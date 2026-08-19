"""
The speech models this project can fine-tune or extract features from.

One registry so that a short name means the same thing in every script:
`--model emotion` selects the same checkpoint in `finetune.run_finetune` and in
`extract.wav2vec`, and each entry carries the layer-dependent defaults that
would otherwise have to be remembered per model.

How much to retrain
-------------------
The default is **freeze the bottom half, adapt the top half** — 12 of 24 layers
for a full backbone, 6 of 12 for the pruned emotion checkpoint — with the CNN
feature extractor always frozen on top of that.

Two facts drive it. The fine-tuning target is 88 eGeMAPS functionals: low-level
acoustic descriptors (F0, loudness, jitter, spectral slopes) whose information
already sits in the early layers, so there is nothing to gain by making those
layers plastic. And the dataset is small — ~25k two-second windows against a
~300M parameter encoder. Unfreeze too much and the network reshapes its whole
representation to emit those 88 numbers, discarding the structure that made the
pretrained features worth using; it then fits eGeMAPS well and predicts the
brain *worse than the frozen baseline*. Always keep the frozen baseline as a
control. `--llrd` is the softer alternative to a hard freeze boundary.

Depth is not the thing to optimise
----------------------------------
"More layers must be better" does not hold here, for two reasons.

*Capacity is not the binding constraint.* Fine-tuning happens on roughly 25k
two-second windows — order 14 hours of audio — against a ~300M parameter
encoder. Data limits the result, not depth; extra capacity mostly buys extra
overfitting risk.

*What layer you read matters more than how many exist.* Paralinguistic and
prosodic information in wav2vec2 peaks in the **middle** layers. The top layers
of `wav2vec2-large-960h` are worse than useless for prosody: that checkpoint is
`Wav2Vec2ForCTC` with `vocab_size=32`, i.e. ASR fine-tuned, so its final layers
have been explicitly optimised to collapse everything except which of 32
characters was spoken — discarding exactly the "how it was said" signal this
project is after. `wav2vec2-large-robust` is `Wav2Vec2ForPreTraining`, the same
24 layers with no such specialisation, which makes it the better 24-layer arm.

The `default_layers` below therefore point mid-network for the *base* models.
Once a model has been fine-tuned here on prosody targets, its upper layers have
been re-tuned toward prosody and the upper range becomes appropriate again — so
extract from `18-23` on our fine-tuned 24-layer checkpoints, not from the base
default. Layer choice is ultimately empirical; `--layers` makes it cheap to
sweep, and it is worth sweeping.

Choosing a comparison set
-------------------------
The obvious experiment — fine-tune `wav2vec2` vs fine-tune `emotion` — is
confounded. The audEERING emotion model is not "wav2vec2-large-960h plus
emotion training"; it differs in three ways at once:

  1. emotion fine-tuning on MSP-Podcast   <- the effect of interest
  2. a different pretraining base (large-robust, not large-960h)
  3. pruned to 12 transformer layers instead of 24

So a difference between those two arms cannot be attributed to emotion
training. `wav2vec2-robust` is in the registry as the control that removes
confound 2: it is the exact base audEERING fine-tuned from. Adding
`--truncate-layers 12` to that arm removes confound 3 as well, leaving emotion
fine-tuning as the only difference.

Suggested arms, cheapest useful set first:

  wav2vec2-robust                       best generic 24-layer arm
  emotion                               the model of interest
  wav2vec2-robust --truncate-layers 12  depth-matched control, only needed to
                                        attribute a difference to emotion
                                        pretraining rather than to depth

If you only run two, run `wav2vec2-robust` and `emotion`: that answers "which
features predict the brain better". Add the truncated arm when you want to say
*why*.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class SpeechModelSpec:
    """A selectable speech encoder and its layer-dependent defaults."""

    key: str
    checkpoint: str
    n_layers: Optional[int]          #: None when only known after loading
    hidden_size: Optional[int]
    default_layers: Optional[str]    #: extraction range, for --layers auto
    note: str = ""

    @property
    def default_freeze(self) -> Optional[int]:
        """Freeze the bottom half; the top half is what fine-tuning adapts.

        Derived from depth rather than stored per entry, so the rule holds for
        every model: 12 of 24 for a full-size backbone, 6 of 12 for the pruned
        emotion checkpoint. See the module docstring for why half.
        """
        return None if self.n_layers is None else self.n_layers // 2

    def describe(self) -> str:
        depth = f"{self.n_layers} layers" if self.n_layers else "depth unknown"
        return (f"{self.key:16s} {self.checkpoint}\n"
                f"{'':16s} {depth}, freeze {self.default_freeze}, "
                f"extract {self.default_layers}\n"
                f"{'':16s} {self.note}")


#: Emotion-pretrained wav2vec2 (audEERING), regressing arousal/dominance/valence.
EMOTION_MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

#: The dimensions its head predicts, in the order the model outputs them.
EMOTION_DIMENSIONS = ["arousal", "dominance", "valence"]

REGISTRY: Dict[str, SpeechModelSpec] = {
    "wav2vec2": SpeechModelSpec(
        key="wav2vec2",
        checkpoint="facebook/wav2vec2-large-960h",
        n_layers=24, hidden_size=1024,
        default_layers="12-17",
        note="ASR fine-tuned (Wav2Vec2ForCTC, vocab_size=32). Its top layers "
             "are specialised for character prediction and discard prosody, "
             "so the default reads mid-network. Prefer wav2vec2-robust.",
    ),
    "wav2vec2-robust": SpeechModelSpec(
        key="wav2vec2-robust",
        checkpoint="facebook/wav2vec2-large-robust",
        n_layers=24, hidden_size=1024,
        default_layers="12-17",
        note="RECOMMENDED 24-layer arm: self-supervised only (no ASR "
             "specialisation) and the exact base audEERING fine-tuned from, so "
             "it doubles as the control for 'emotion'. Add --truncate-layers 12 "
             "to match its depth as well.",
    ),
    "emotion": SpeechModelSpec(
        key="emotion",
        checkpoint=EMOTION_MODEL,
        n_layers=12, hidden_size=1024,
        default_layers="6-11",
        note="Encoder already shaped by an affective-prosody objective on "
             "MSP-Podcast, so its UPPER layers are the affect-tuned ones and "
             "the default reads high. PRUNED TO 12 LAYERS — layer settings for "
             "a 24-layer model are invalid here.",
    ),
    "hubert": SpeechModelSpec(
        key="hubert",
        checkpoint="facebook/hubert-large-ll60k",
        n_layers=24, hidden_size=1024,
        default_layers="12-17",
        note="Alternative self-supervised backbone, no ASR specialisation.",
    ),
    "wavlm": SpeechModelSpec(
        key="wavlm",
        checkpoint="microsoft/wavlm-large",
        n_layers=24, hidden_size=1024,
        default_layers="12-17",
        note="Speaker/overlap-aware pretraining; often strong on paralinguistics.",
    ),
}

#: Back-compat alias used by older call sites.
MODEL_MAP = {key: spec.checkpoint for key, spec in REGISTRY.items()}


def resolve_model(name: str) -> SpeechModelSpec:
    """Look up a registry key, or wrap an arbitrary checkpoint id.

    Anything not in the registry is treated as a Hugging Face id or a local
    directory. Its depth is unknown until the weights load, so the
    layer-dependent defaults are left as None and must be given explicitly.
    """
    if name in REGISTRY:
        return REGISTRY[name]
    return SpeechModelSpec(
        key=name, checkpoint=name,
        n_layers=None, hidden_size=None,
        default_layers=None,
        note="Custom checkpoint; pass --freeze-layers and --layers explicitly.",
    )


def resolve_base_model(model_type: str, base_model_name: Optional[str] = None) -> str:
    """Checkpoint for a registry key, or an explicit override."""
    if base_model_name:
        return base_model_name
    if model_type not in REGISTRY:
        raise ValueError(
            f"Unknown model {model_type!r}; choose from {sorted(REGISTRY)} "
            f"or pass an explicit checkpoint."
        )
    return REGISTRY[model_type].checkpoint


def format_registry() -> str:
    return "\n\n".join(spec.describe() for spec in REGISTRY.values())

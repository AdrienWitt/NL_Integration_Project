import os
import numpy as np
from os.path import join, dirname

from .textgrid import TextGrid


def load_textgrids(stories, textgrid_dir=None):
    """Parse the word/phoneme TextGrid of each story.

    Parameters
    ----------
    stories : list of str
        Story names, without the ``.TextGrid`` suffix.
    textgrid_dir : str or Path, optional
        Directory holding ``<story>.TextGrid``. Defaults to
        ``config.TEXTGRID_DIR``.

    Returns
    -------
    dict of {story: TextGrid}
    """
    if textgrid_dir is None:
        from config import TEXTGRID_DIR
        textgrid_dir = TEXTGRID_DIR

    grids = {}
    for story in stories:
        grid_path = os.path.join(str(textgrid_dir), "%s.TextGrid" % story)
        with open(grid_path) as f:
            grids[story] = TextGrid(f.read())
    return grids

class TRFile(object):
    def __init__(self, trfilename, expectedtr=2.0045):
        """Loads data from [trfilename], should be output from stimulus presentation code.
        """
        self.trtimes = []
        self.soundstarttime = -1
        self.soundstoptime = -1
        self.otherlabels = []
        self.expectedtr = expectedtr
        
        if trfilename is not None:
            self.load_from_file(trfilename)
        

    def load_from_file(self, trfilename):
        """Loads TR data from report with given [trfilename].
        """
        ## Read the report file and populate the datastructure
        for ll in open(trfilename):
            timestr = ll.split()[0]
            label = " ".join(ll.split()[1:])
            time = float(timestr)

            if label in ("init-trigger", "trigger"):
                self.trtimes.append(time)

            elif label=="sound-start":
                self.soundstarttime = time

            elif label=="sound-stop":
                self.soundstoptime = time

            else:
                self.otherlabels.append((time, label))
        
        ## Fix weird TR times
        itrtimes = np.diff(self.trtimes)
        badtrtimes = np.nonzero(itrtimes>(itrtimes.mean()*1.5))[0]
        newtrs = []
        for btr in badtrtimes:
            ## Insert new TR where it was missing..
            newtrtime = self.trtimes[btr]+self.expectedtr
            newtrs.append((newtrtime,btr))

        for ntr,btr in newtrs:
            self.trtimes.insert(btr+1, ntr)

    def simulate(self, ntrs):
        """Simulates [ntrs] TRs that occur at the expected TR.
        """
        self.trtimes = list(np.arange(ntrs)*self.expectedtr)
    
    def get_reltriggertimes(self):
        """Returns the times of all trigger events relative to the sound.
        """
        return np.array(self.trtimes)-self.soundstarttime

    @property
    def avgtr(self):
        """Returns the average TR for this run.
        """
        return np.diff(self.trtimes).mean()

def load_simulated_trfiles(respdict, tr=2.0, start_time=10.0, pad=5):
    trdict = dict()
    for story, resps in respdict.items():
        trf = TRFile(None, tr)
        trf.soundstarttime = start_time
        trf.simulate(resps - pad)
        trdict[story] = [trf]
    return trdict

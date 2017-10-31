from core.data.calcium_imaging import JaneliaData
import matplotlib.pyplot as plt
import sys
import cPickle as pickle

def key_down(event):
        # quit the application by pressing escape
        if event.key == "escape":
            sys.exit(0)

        # only continue if 1, 2, 3 or 4 is pressed
        if event.key not in ["1", "2", "3", "4"]:
            return
        k = int(event.key)

        if k == 1:
            label = True
            print "Marked okay"
        elif k == 2:
            label = False
            print "Marked shitty"

        sub_quality[sub.name] = label
        # close the current figure (and show the next one)
        plt.close()

if __name__ == "__main__":
    jd = JaneliaData()
    sub_quality = {}
    for sub in jd.subvolumes:
        fig = plt.figure()
        ax=fig.add_subplot(111)
        ax=sub.plot_activity_labels(labels=('active_very_certain', 'active_mod_certain', 'uncertain'), cmap='gray', ax=ax)
        fig.canvas.mpl_connect("key_press_event", key_down)
        plt.show()

    pickle.dump(sub_quality, open('out/sub_quality.p', 'w'))
    print 'Done.'


""" OUTPUT
sub_quality = { 'an197522_2013_03_08_01002': True, 'an197522_2013_03_08_01003': True, 'an197522_2013_03_08_01004': True,
            'an197522_2013_03_08_02002': True, 'an197522_2013_03_08_02003': True, 'an197522_2013_03_08_02004': True,
            'an197522_2013_03_08_03004': True, 'an197522_2013_03_08_05004': True, 'an197522_2013_03_08_06003': True,
            'an197522_2013_03_08_06004': True, 'an197522_2013_03_08_07003': True, 'an197522_2013_03_08_08002': True,
            'an197522_2013_03_08_08003': False, 'an197522_2013_03_08_08004': False, 'an197522_2013_03_08_09002': True,
            'an197522_2013_03_08_09003': False, 'an197522_2013_03_08_09004': False, 'an197522_2013_03_08_10002': False,
            'an197522_2013_03_08_10003': False, 'an197522_2013_03_08_10004': False, 'an197522_2013_03_10_11002': False,
            'an197522_2013_03_10_11003': False, 'an197522_2013_03_10_11004': False, 'an197522_2013_03_10_12002': True,
            'an197522_2013_03_10_13002': True, 'an197522_2013_03_10_14003': True, 'an197522_2013_03_10_14004': True,
            'an197522_2013_03_10_15002': True, 'an197522_2013_03_10_15003': False, 'an197522_2013_03_10_15004': True,
            'an197522_2013_03_10_16002': True, 'an197522_2013_03_10_16003': True, 'an197522_2013_03_10_16004': False,
            'an197522_2013_03_10_17002': False, 'an197522_2013_03_10_17003': False, 'an197522_2013_03_10_17004': False,
            'an197522_2013_03_10_18002': False, 'an197522_2013_03_10_18003': False, 'an197522_2013_03_10_18004': False,
            'an197522_2013_03_10_19002': False, 'an197522_2013_03_10_19003': False, 'an229717_2013_12_01_01003': False,
            'an229717_2013_12_01_01004': True, 'an229717_2013_12_01_02002': False, 'an229717_2013_12_01_02003': False,
            'an229717_2013_12_01_02004': False, 'an229717_2013_12_01_04002': True, 'an229717_2013_12_01_04003': True,
            'an229717_2013_12_01_07002': True, 'an229717_2013_12_01_07003': True, 'an229717_2013_12_01_07004': True,
            'an229719_2013_12_02_06002': True, 'an229719_2013_12_02_09002': False, 'an229719_2013_12_02_09003': True,
            'an229719_2013_12_02_09004': False, 'an229719_2013_12_02_10002': True, 'an229719_2013_12_02_12002': False,
            'an229719_2013_12_02_12003': False, 'an229719_2013_12_02_12004': True, 'an229719_2013_12_02_15002': True,
            'an229719_2013_12_02_15003': False, 'an229719_2013_12_02_15004': False, 'an229719_2013_12_05_02002': False,
            'an229719_2013_12_05_02003': True, 'an229719_2013_12_05_02004': True, 'an229719_2013_12_05_03002': True,
            'an229719_2013_12_05_03003': False, 'an229719_2013_12_05_03004': True, 'an229719_2013_12_05_05003': True,
            'an229719_2013_12_05_05004': True, 'an229719_2013_12_05_06002': True, 'an229719_2013_12_05_06003': True,
            'an229719_2013_12_05_06004': True, 'an229719_2013_12_05_07002': True, 'an229719_2013_12_05_07003': True,
            'an229719_2013_12_05_07004': True, 'an229719_2013_12_05_08002': True, 'an229719_2013_12_05_08003': True,
            'an229719_2013_12_05_08004': False}
"""
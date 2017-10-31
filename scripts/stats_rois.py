import re
from core.data.calcium_imaging import JaneliaData

jd = JaneliaData()
roi_ids = []
for _sub in jd.subvolumes:
    for _roi in _sub.rois:
        roi_ids.append(_roi.name.split('(')[0])

ids_per_session = [re.sub(r'_\d\d\d\d\d_', '_', roi_ids[i]) for i in xrange(len(roi_ids))]
print 'Overall number of Roi objects found in %d subvolumes: %d' % (len(jd.subvolumes), len(ids_per_session))
print 'Number of unique neurons over all sessions:', len(set(ids_per_session))

"""
OUTPUT:
Overall number of Roi objects found in 79 subvolumes: 26036
Number of unique neurons over all sessions: 25152
"""
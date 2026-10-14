"""Cyclone positions and cell events; TE extraction is a separate verified step.

Coordinates are (latitude, longitude) degrees. Unit-vector means handle the
dateline. Heatmap cells use integer-degree edges, latitude [-90,90] and longitude
[0,360); the north pole belongs to the last latitude cell. These implementation
conventions must accompany reports because paper prose does not specify them.
"""
import csv
import numpy as np
from scipy.optimize import linear_sum_assignment
from rev import confusion


def distance_km(a,b):
    a=np.deg2rad(np.asarray(a,dtype=float));b=np.deg2rad(np.asarray(b,dtype=float))
    d=b-a
    h=np.sin(d[...,0]/2)**2+np.cos(a[...,0])*np.cos(b[...,0])*np.sin(d[...,1]/2)**2
    return 6371.0088*2*np.arcsin(np.sqrt(np.clip(h,0,1)))


def mean_position(points):
    values=np.asarray(points,dtype=float)
    if values.ndim!=2 or values.shape[1]!=2 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Invalid cyclone positions')
    lat,lon=np.deg2rad(values).T
    xyz=np.array([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)]).mean(axis=1)
    if np.linalg.norm(xyz)<1e-12:raise ValueError('Undefined spherical mean')
    return np.array([np.rad2deg(np.arctan2(xyz[2],np.hypot(*xyz[:2]))),
                     np.rad2deg(np.arctan2(xyz[1],xyz[0]))%360])


def read_tracks(path):
    tracks={}
    with open(path) as stream:
        for row in csv.DictReader(stream,skipinitialspace=True):
            stamp=np.datetime64(f"{int(row['year']):04d}-{int(row['month']):02d}-{int(row['day']):02d}T{int(row['hour']):02d}",'ns')
            point=np.array([float(row['lat']),float(row['lon'])])
            if not np.isfinite(point).all() or not -90<=point[0]<=90:raise ValueError('Invalid track point')
            track=tracks.setdefault(row['track_id'],{})
            if stamp in track:raise ValueError('Duplicate track timestamp')
            track[stamp]=point
    return tracks


def match_initial(tracks,truth,initialization,radius_km=100):
    """One-to-one, maximum-cardinality matching; minimize distance on ties.

Only time-zero positions influence identity. Never rematch using future truth.
Unmatched dummies cost enough to prefer any admissible match; forbidden edges
cost more than all dummies combined.
"""
    stamp=np.datetime64(initialization,'ns')
    ids=sorted(k for k,v in tracks.items() if stamp in v)
    storms=sorted(k for k,v in truth.items() if stamp in v)
    if not ids or not storms:return {}
    distances=distance_km(np.array([tracks[k][stamp] for k in ids])[:,None,:],
                          np.array([truth[k][stamp] for k in storms])[None,:,:])
    unmatched=(len(ids)+1)*(radius_km+1)
    costs=np.full((len(ids),len(storms)+len(ids)),unmatched)
    costs[:,:len(storms)]=np.where(distances<=radius_km,distances,unmatched*(len(ids)+1))
    rows,cols=linear_sum_assignment(costs)
    return {storms[c]:ids[r] for r,c in zip(rows,cols) if c<len(storms) and distances[r,c]<=radius_km}


def position_rows(members,truth,initialization,lead_hours,minimum_fraction=.5):
    if len(members)<2:raise ValueError('Ensemble required')
    init=np.datetime64(initialization,'ns')
    matches=[match_initial(tracks,truth,init) for tracks in members]
    rows=[]
    for storm,observations in sorted(truth.items()):
        for lead in lead_hours:
            stamp=init+np.timedelta64(lead,'h')
            if stamp not in observations:continue
            points=[tracks[m[storm]][stamp] for tracks,m in zip(members,matches)
                    if storm in m and stamp in tracks[m[storm]]]
            if len(points)<len(members)*minimum_fraction:continue
            position=mean_position(points)
            rows.append(dict(storm=storm,lead_hours=lead,members_present=len(points),
                mean_position=position.tolist(),truth_position=np.asarray(observations[stamp]).tolist(),
                error_km=float(distance_km(position,observations[stamp]))))
    return rows


def paired_position_rows(baseline,candidate):
    def index(rows):
        result={(r['storm'],r['lead_hours']):r for r in rows}
        if len(result)!=len(rows):raise ValueError('Duplicate cyclone/lead pair')
        return result
    a=index(baseline);b=index(candidate);rows=[]
    for key in sorted(a.keys()&b.keys()):
        if a[key]['truth_position']!=b[key]['truth_position']:raise ValueError('Changed cyclone truth')
        rows.append(dict(storm=key[0],lead_hours=key[1],baseline_km=a[key]['error_km'],
            candidate_km=b[key]['error_km'],difference_km=b[key]['error_km']-a[key]['error_km']))
    return rows


def cell_events(tracks,stamp):
    events=np.zeros((180,360),dtype=bool)
    stamp=np.datetime64(stamp,'ns')
    for track in tracks.values():
        if stamp not in track:continue
        lat,lon=track[stamp]
        if not np.isfinite([lat,lon]).all() or not -90<=lat<=90:raise ValueError('Invalid cell location')
        events[min(int(np.floor(lat+90)),179),int(np.floor(lon%360))]=True
    return events


def cell_confusion(members,truth,stamp):
    forecast=np.stack([cell_events(tracks,stamp) for tracks in members])
    observed=cell_events(truth,stamp)
    edges=np.deg2rad(np.arange(-90,91))
    area=np.diff(np.sin(edges))[:,None]*np.ones((1,360))
    return confusion(forecast,observed,np.full(observed.shape,.5),area)

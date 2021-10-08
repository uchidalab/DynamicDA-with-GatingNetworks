import numpy as np
np.random.seed(1111)

def noise(x): # return random numbers
    x = np.random.normal(loc=0., scale=0.5, size=len(x)).tolist()
    return x

def identity(x):
    return x

def jitter(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    x = [t+np.random.normal(loc=0., scale=sigma, size=None) for t in x]
    return np.array(list(x))

def scaling(x, sigma=0.2):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=1., scale=sigma, size=None)
    x = [np.multiply(t, factor) for t in x]
    return np.array(list(x))

def rotation(x): # this is flipping
    flip = np.random.choice([-1, 1], size=None) #ランダムに-1か1を取り出す
    x = [flip*i for i in x]
    return x

def permutation(x, max_segments=5, seg_mode="equal"):
    orig_steps = np.arange(len(x))
    num_segs = np.random.randint(1, max_segments, size=None)
    
    ret = np.zeros_like(x)
    if num_segs > 1:
        if seg_mode == "random":
            split_points = np.random.choice(len(x)-2, num_segs-1, replace=False)
            split_points.sort()
            splits = np.split(orig_steps, split_points)
        else:
            splits = np.array_split(orig_steps, num_segs)
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        rng = np.random.default_rng()
        warp = np.concatenate(rng.permutation(splits)).ravel()
        np_x = np.array([x]).T
        ret = np_x[warp]
        ret = [float(i) for i in ret]
    else:
        ret = x
    return ret

def magnitudeWarp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(len(x))
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot+2))
    warp_steps = (np.ones((1))*(np.linspace(0, len(x)-1., num=knot+2))).T
    
    ret = np.zeros_like(x)
    warper = np.array([CubicSpline(warp_steps[:], random_warps[:])(orig_steps)]).T
    ret = x * warper[:,0]
    return ret

def timeWarp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(len(x))
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot+2))
    warp_steps = (np.ones((1))*(np.linspace(0, len(x)-1., num=knot+2))).T
    
    ret = np.zeros_like(x)
    time_warp = CubicSpline(warp_steps[:], warp_steps[:] * random_warps[:])(orig_steps)
    scale = (len(x)-1)/time_warp[-1]
    ret = np.interp(orig_steps, np.clip(scale*time_warp, 0, len(x)-1), x).T
    return ret

def windowSlice(x, reduce_ratio=0.9):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    target_len = np.ceil(reduce_ratio*len(x)).astype(int)
    if target_len >= len(x):
        return x
    starts = np.random.randint(low=0, high=len(x)-target_len, size=None)
    ends = (target_len + starts).astype(int)
    
    ret = np.zeros_like(x)
    ret = np.interp(np.linspace(0, target_len, num=len(x)), np.arange(target_len), x[starts:ends]).T
    return ret

def windowWarp(x, window_ratio=0.1, scales=[0.5, 2.]):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = np.random.choice(scales)
    warp_size = np.ceil(window_ratio*len(x))
    window_steps = np.arange(warp_size)
        
    window_starts = np.random.randint(low=1, high=len(x)-warp_size-1, size=None)
    window_ends = (window_starts + warp_size).astype(int)
    
    ret = np.zeros_like(x)
    start_seg = x[:window_starts]
    window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales)), window_steps, x[window_starts:window_ends])
    end_seg = x[window_ends:]
    warped = np.concatenate((start_seg, window_seg, end_seg))
    ret = np.interp(np.arange(len(x)), np.linspace(0, len(x)-1., num=warped.size), warped).T
    return ret

def toy1(x, idx):
    if idx%2==0: # for even numbers
        x = x
    else:
        x = np.random.normal(loc=0., scale=0.5, size=len(x)).tolist()
    return x
    
def toy2(x, idx):
    if idx%2!=0: # for odd numbers
        x = x
    else:
        x = np.random.normal(loc=0., scale=0.5, size=len(x)).tolist()
    return x


if __name__=="__main__":
    x = np.arange(0,1000,2)
    magnitudeWarp(x)

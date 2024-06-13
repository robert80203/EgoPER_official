import numpy as np

class ScaleDown():
    def __init__(self, H, W, target=256):
        self.H = H
        self.W = W
        self.target = target
        
        self.ratio = self.target / min([H, W])
        
        self.output_h = int(np.round(H * self.ratio)) 
        self.output_w = int(np.round(W * self.ratio)) 
        
        
    def downsample(self, h, w):
        h_ = int(np.round(h * self.ratio))
        w_ = int(np.round(w * self.ratio))

        return h_, w_

class CentralCrop():
    def __init__(self, H, W, target=224):
        self.H = H
        self.W = W
        self.target = target

        self.h_head_pad = (H - target) // 2
        self.w_head_pad = (W - target) // 2
        
        
        self.output_h = target
        self.output_w = target
        
        
    def downsample(self, h, w):
        h_ = max( 0, min( self.target-1, h - self.h_head_pad ) )
        w_ = max( 0, min( self.target-1, w - self.w_head_pad ) )

        return h_, w_

def read_annotation(annot_fname):
    """
    parse the annotation file
    """
    with open(annot_fname) as fp:
        content = fp.read()

        annot_lines = content.split('\n')[:-1]
        annot_list = []
        for annot in annot_lines:
            annot = [ float(x) for x in annot.split(',') ]
            annot_list.append(annot)

    return annot_list

def generate_hand_mask(annot_list, trans_list, final_size):
    """
    return a mask matrix of shape (5 x final_size[0] x final_size[1]) 
    5 channels store the information of hand_score, NC, SC, PC, OC
    each channel is generated as:
        1. create a 0-1 mask according to bbox.
        2. multiply the mask with one of the scores, e.g. mask = mask * hand_score.
    """
    if isinstance(final_size, int):
        final_size = [ final_size, final_size ]
    
    if len(annot_list) == 0:
        return np.zeros([5] + final_size, dtype=np.float32)

    def translate_cord(h, w):
        for trans in trans_list:
            h, w = trans.downsample(h, w)
        return h, w

    # generate a mask for each hand
    M = []
    for annot in annot_list:
        mask = np.zeros(final_size, dtype=np.float32)
        
        l, t, r, b = map(int, annot[1:5])
        t, l = translate_cord(t, l)
        b, r = translate_cord(b, r)
        
        mask[ t:b, l:r ] = 1 # h, w
        M.append(mask)
    
    downsample = np.stack(M, axis=-1) # H, W, N
    
    div = downsample.sum(2) + 1e-5 # h, w
    small = []
    for j in [0, 5, 6, 7, 8]:
        val = [ a[j] for a in annot_list ] # N
        mask = (downsample * val).sum(2) / div
        small.append(mask)
    small = np.stack(small, axis=0) # 5 x h x w

    return small

if __name__ == '__main__':
    """
    example:
        1.take a hand detection file
        2.create a ReshapeCordMap with the W, H of original image 
          and the downsample (256) and central crop (224) size 
          to help remove out-of-crop region
        3.generate mask
    """
    annot_fname = '...'
    annot_list = read_annotation(annot_fname)
    s = ScaleDown(720, 1280, target=256)
    c = CentralCrop(s.output_h, s.output_w, target=224)
    s2 = ScaleDown(c.output_h, c.output_w, target=7)
    mask = generate_hand_mask(annot_list, [s, c, s2], 7)
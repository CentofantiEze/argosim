import numpy as np


def shift_beam(beam, shift_x, shift_y):
    beam_shift = np.roll(beam,shift_x,axis=1)
    if shift_x<0:
        beam_shift[:,shift_x:] = 0
    else:
        beam_shift[:,:shift_x] = 0

    beam_shift = np.roll(beam_shift,shift_y,axis=0)
    if shift_y<0:
        beam_shift[shift_y:,:] = 0
    else:
        beam_shift[:shift_y,:] = 0
    return beam_shift

def find_peak(I):
    y_max, x_max = int(np.argmax(I)/I.shape[0]), np.mod(np.argmax(I),I.shape[0])
    x_off, y_off = int(I.shape[0]/2), int(I.shape[1]/2)
    shift_x, shift_y = x_max-x_off,y_max-y_off
    max_val = np.max(I)
    return max_val, x_max, y_max, shift_x, shift_y

def clean_beam(B, search_box=20):
    max_val, x_max, y_max, shift_x, shift_y = find_peak(B)
    B_clean = B[y_max-search_box:y_max+search_box,x_max-search_box:x_max+search_box]
    B_clean = np.pad(B_clean,((y_max-search_box,B.shape[0]-y_max-search_box),(x_max-search_box,B.shape[1]-x_max-search_box)))
    return B_clean
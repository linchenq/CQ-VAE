import os
import numpy as np
import tqdm

def recursive(remain, sumed, step, out=[]):
    if remain == 1:
        out.append(sumed)
        yield out
        
    else:
        for i in range(0, sumed + 1, step):
            yield from recursive(remain-1, sumed-i, step, out + [i])
    
    
if __name__ == '__main__':
    '''
    Trick: for linear combination, we'd like to see a+b+c+... = 1
                if step size is set to 0.1, multiply each by 10
                if step size is set to 0.01, multiply each by 100
    '''
    out = []
    rate = 0.01
    remain = 3
    sumed = 100
    step = 5
    rets = recursive(remain=remain, sumed=sumed, step=step)
    
    for ret in tqdm.tqdm(rets):
        ret = [round(ele * rate, 3) for ele in ret]
        out.append(ret)
    
    np.save("cfgs_table.npy", np.array(out))
    print(f"the whole length is {len(out)}")
    
            
        
        
        
    
    
    
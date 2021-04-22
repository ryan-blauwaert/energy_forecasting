import numpy as np
import pandas as pd


if __name__ == '__main__':
    l1 = [1,2,3,4,5]
    l2 = ['a', 'b', 'c', 'd', 'e']
    df = pd.DataFrame(l1, columns=['Time'])
    df['Megawatthours'] = l2
    print(df)
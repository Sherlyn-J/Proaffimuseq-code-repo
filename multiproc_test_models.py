import multiprocessing as mp
import pandas as pd
import numpy as np
import sys
import time

from sklearn.model_selection    import train_test_split, cross_val_predict
from sklearn.preprocessing      import MinMaxScaler
from sklearn.metrics            import mean_absolute_error as mae
from scipy.stats                import pearsonr as pcc

#methods
from sklearn.linear_model       import LinearRegression
from sklearn.neural_network     import MLPRegressor
from sklearn.svm                import SVR, NuSVR, LinearSVR
from sklearn.neighbors          import RadiusNeighborsRegressor, KNeighborsRegressor
from sklearn.ensemble           import ExtraTreesRegressor, RandomForestRegressor

# output queue
output = mp.Queue()

# test models
def run_job( file, proc_id ):
    """ Testing feature sets """

    data = pd.read_csv( file )
    df = pd.DataFrame( data )
    X = df.iloc[:,1:-1]
    y = df.iloc[:,-1]
    columns = X.columns
    #standardize X
    std = MinMaxScaler()
    X = std.fit_transform( X )

	# start with
    R, MAE = 0, 1
 
    #iterate n-features
    for i in range( 1, max_features+1 ): # limit features to prevent overfit
        for ndx in [ selected_ndx + [ e ] for e in range(n) if e not in selected_ndx ]: # n=total features
            #Cross validate
            nX = X[:,list(ndx)]
            learner = RandomForestRegressor()
            y_pred  = cross_val_predict( learner, nX, y, cv=10 ) # 10-fold cv
            R_new, MAE_new = pcc( y, y_pred )[0], mae( y, y_pred )
            if R_new > R and MAE_new < MAE:
                R, MAE, selected_ndx = R_new, MAE_new, ndx

				
if __name__ == '__main__':
           
    for proc_id in range(total_processes):
        # list of processes to run
        processes = [mp.Process(target=run_job, args=( 'path/filename', proc_id )) ]

        # Start processes
        for p in processes:
            p.start()

        # Complete and exit processes
        for p in processes:
            p.join()

#---------------------------------------------------------------
#---------------------------------------------------------------
import numpy as np
#!/usr/bin/env python
# On 20130210, v0.2
# Critical Line Algorithm
# by MLdP <lopezdeprado@lbl.gov>
#---------------------------------------------------------------
#---------------------------------------------------------------
import cProfile, pstats,StringIO
import time
import pycuda.driver as drv
import pycuda.autoinit
from   pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
#################################################CUDA KERNELS ############################
mod = SourceModule('''
/*
__global__ void column_append(float* matrix, float* column,int numCols, int col )
{
	//compute the threadposition
	int index = blockIdx.x*numCols+col;
	//insert column value into matrix at index position
	matrix[index] = column[blockIdx.x];
}
*/
__global__ void column_append(float* matrix, float* column, int numThreads, int col)
{
	//compute threadposition
	int index = numThreads*threadIdx.x+ col;
	//insert column values into matrix at computed index position
	matrix[index] = column[threadIdx.x];

}

__global__ void row_append(float* matrix, float* row, int row_pos)
{
	//compute the threadposition
	int index =blockDim.x*row_pos+threadIdx.x;
	//insert row value into matrix at index position
	matrix[index] = row[threadIdx.x];
}
''')
##########################################################################################
#intialize function modules?
column_append = mod.get_function("column_append")
row_append = mod.get_function("row_append")

def extract_columns(mat, start=0, stop=None):
    "code by Hannes Bretschneider : www.hannes.brt.github.io"
    dtype = mat.dtype
    itemsize = np.dtype(dtype).itemsize
    N, M = mat.shape
    m = stop - start
    assert mat.flags.c_contiguous
    assert start >= 0 and start <= M and stop >= 0 and stop <= M and stop > start
    new_mat = gpuarray.empty((N, m), dtype)
    copy = drv.Memcpy2D()
    copy.set_src_device(mat.gpudata)
    copy.src_x_in_bytes = start * itemsize    # Offset of the first column in bytes
    copy.set_dst_device(new_mat.gpudata)
    copy.src_pitch = M * itemsize   # Width of a row in bytes in the source array
    copy.dst_pitch = copy.width_in_bytes = m * itemsize  # Width of sliced row
    copy.height = N
    copy(aligned=True)
    return new_mat
#---------------------------------------------------------------    
def initAlgo(data_mean,lower_bound, upper_bound):
    # Initialize the algo
    #1) Form structured array
    a=np.zeros((data_mean.shape[0]),dtype=[('id',int),('mu',float)])
    b=[data_mean[i][0] for i in range(data_mean.shape[0])] # dump array into list
    a[:]=zip(range(data_mean.shape[0]),b) # fill structured array
    #2) Sort structured array
    b=np.sort(a,order='mu')
    #3) First free weight
    i,w=b.shape[0],np.copy(lower_bound)
    while sum(w)<1:
        i-=1
        w[b[i][0]]=upper_bound[b[i][0]]
    w[b[i][0]]+=1-sum(w)
#    print  [b[i][0]]
#    c = [b[i][0]]
#    print "value stored in c: {}".format(c)	
#    print "length: {}".format(len(c)) 
#    raw_input("system_pause") 	
    return [b[i][0]],w
#---------------------------------------------------------------
def getB(data_mean, f):
    return diffLists(range(data_mean.shape[0]),f)
#---------------------------------------------------------------
def diffLists(list1,list2):
    return list(set(list1)-set(list2))
#---------------------------------------------------------------
def reduceMatrix_cpu(input_matrix, listX, listY):
	if len(listX)==0 or len(listY)==0:return
	intermediate_matrix=input_matrix[:,listY[0]:listY[0]+1]
	for i in listY[1:]:
		a=input_matrix[:,i:i+1]
		intermediate_matrix=np.append(intermediate_matrix,a,1)
	return_matrix=intermediate_matrix[listX[0]:listX[0]+1,:]
	for i in listX[1:]:
		a=intermediate_matrix[i:i+1,:]
		return_matrix=np.append(return_matrix, a, 0)
	print return_matrix
	raw_input("systemPause")
	return return_matrix
#---------------------------------------------------------------
def reduceMatrix_gpu(input_matrix,listX,listY):

    #if either lists are empty then end function call	
    if len(listX)==0 or len(listY)==0:return
#    print "shape of input matrix: {}\n".format(input_matrix.shape)	
#    print "input_matrix: \n {} \n ".format(input_matrix)
#    print "lenghts of input lists x: {}, y: {}\n".format(len(listX), len(listY))
#    print "listX: \n {} \n".format(listX)
#    print "listY: \n {} \n".format(listY)
    
    #create a return matrix (gpuarray) of shape (listX, listY)
    ret_x = len(listX)
    ret_y = len(listY)	

    if (ret_x*ret_y) > 25:
	gpu_return_matrix = gpuarray.to_gpu(np.ones((ret_x,ret_y)).astype(np.float32))
	#create return matrix oldschool way
	#return_matrix = np.zeros((ret_x,ret_y)).astype(np.float32)
	#gpu_ret_matrix = cuda.mem_alloc(return_matrix.nbytes)
		
    	#create an intermediate matrix (gpuarray) of shape ((input_matrix.shape[0]),len(listY))
        inter_x = input_matrix.shape[0]
	inter_y = len(listY)
    	gpu_intermediate_matrix = gpuarray.to_gpu(np.ones((inter_x,inter_y)).astype(np.float32))

    	#create intermediate matrix oldschool way
#    	intermediate_matrix = np.zeros((inter_x, inter_y)).astype(np.float21)
#    	gpu_intermediate_matrix = cuda.mem_alloc(intermediate_matrix.nbytes)	
#    	cuda.memcpy_htod(gpu_intermediate_matrix,intermediate_matrix)
	
    	num_cols = (len(listY))
    	num_rows = (len(listY))		

    	#for every value(integer) in the list
    	j = 0 
    	num_of_blocks =gpu_intermediate_matrix.shape[0]
    	num_of_rows = gpu_intermediate_matrix.shape[0]
	
    	
#    	num_of_blocks =intermediate_matrix.shape[0]
#    	num_of_rows = intermediate_matrix.shape[0]

#    	print "number of blocks:{}\n ".format(num_of_blocks)
#    	print "number of columns: {}\n".format(num_cols)       
    	for i in listY[0:]:
#		print "the position of the matix to be appended: {} ".format(j)
		a = gpuarray.to_gpu(input_matrix[:,i:i+1].astype(np.float32))
#		print "The value pulled from original column variable \n {} \n ".format(old_a)
#		print "The value pulled from new column variable \n {} \n ".format(a)
#		print "the shape of a: {} \n".format(a.shape)
#		print"the value of a \n{}\n".format(a)
		#copy the column(slice the input array column wise) and store into variable
#        	column_append(gpu_intermediate_matrix, a, np.int32(num_cols), np.int32(j)\
#				,block=(1,1,1), grid=(num_of_blocks,1,1))	
		column_append( gpu_intermediate_matrix, a, np.int32(num_rows), np.int32(j)\
			, block=(num_of_rows,1,1), grid=(1,1,1))
		j = j+1
#		print "intermediate matrix inside for loop: \n {} \n".format(gpu_intermediate_matrix)
    	j = 0
    	#for every ith value(integer) in the xlist, slice the rows of the intermediate matrix
    	for i in listX[0:]:
#		print "the positino of the matrix to be appended: {}".format(j)
		a = gpu_intermediate_matrix[i:i+1,:]
#		print "the shape of b: {} \n".format(a.shape)
#		print"the value of b: \n{}\n".format(a)
		#copy a row from the intermediate matrix
        	row_append(gpu_return_matrix, gpu_intermediate_matrix[i:i+1,:], np.int32(j),\
				block=(num_rows,1,1), grid=(1,1,1))
 		j = j+1

    	#return the solution matrix (should be shape (len(listX,listY))
    	return_matrix = gpu_return_matrix.get()
#    	cuda.memcpy_dtoh(return_matrix, gpu_return_matrix)	
#    	print "intermediate matrix shape: {} \n".format(gpu_intermediate_matrix.shape)
#    	print "intermediate matrix: \n {} \n".format(gpu_intermediate_matrix)
#    	print "return matrix shape: {} \n".format(return_matrix.shape)	
#    	print "return matrix: \n {} \n".format(return_matrix)
#    	raw_input("system_pause")
#	print "large matrix \n"
#    	if (ret_x*ret_y) > 25:
#		print"intermediate matrix size: {} \n".format(gpu_intermediate_matrix.shape)
#    		print"return_matrix_size: {} and shape {} \n".format((ret_x*ret_y), return_matrix.shape)
#		print "values in return_matrix: {} \n".format(return_matrix)
#		raw_input("system pause this")	    
    else:
    	intermediate_matrix=input_matrix[:,listY[0]:listY[0]+1]
    	for i in listY[1:]:
        	a=input_matrix[:,i:i+1]
        	intermediate_matrix=np.append(intermediate_matrix,a,1)
    	return_matrix=intermediate_matrix[listX[0]:listX[0]+1,:]
    	for i in listX[1:]:
        	a=intermediate_matrix[i:i+1,:]
        	return_matrix=np.append(return_matrix,a,0)
    print return_matrix
    raw_input("systempause")	
    return return_matrix
#---------------------------------------------------------------    
def purgeNumErr(free_weights ,gammas ,lambdas,lower_bound,upper_bound,solution_set,tol):
    # Purge violations of inequality constraints (associated with ill-conditioned covar matrix)
    i=0
    while True:
        flag=False
        if i==len(solution_set):break
        if abs(sum(solution_set[i])-1)>tol:
            flag=True
        else:
            for j in range(solution_set[i].shape[0]):
                if solution_set[i][j]-lower_bound[j]<-tol or solution_set[i][j]-upper_bound[j]>tol:
                    flag=True;break
        if flag==True:
            del solution_set[i]
            del lambdas[i]
            del gammas[i]
            del free_weights[i]
        else:
            i+=1
    return
#---------------------------------------------------------------    
def purgeExcess(data_mean, lambdas, gammas, free_weights, solution_set):
    # Remove violations of the convex hull
    i,repeat=0,False
    while True:
        if repeat==False:i+=1
        if i==len(solution_set)-1:break
        w=solution_set[i]
        mu=np.dot(w.T,data_mean)[0,0]
        j,repeat=i+1,False
        while True:
            if j==len(solution_set):break
            w=solution_set[j]
            mu_=np.dot(w.T,data_mean)[0,0]
            if mu<mu_:
                del solution_set[i]
                del lambdas[i]
                del gammas[i]
                del free_weights[i]
                repeat=True
                break
            else:
                j+=1
    return
#---------------------------------------------------------------
def getMinVar(data_covar, solution_set):
    # Get the minimum variance solution
    var=[]
    for w in solution_set:
        a=np.dot(np.dot(w.T,data_covar),w)
        var.append(a)
    return min(var)**.5,solution_set[var.index(min(var))]
#---------------------------------------------------------------
def getMaxSR( data_mean,data_covar,solution_set):
    # Get the max Sharpe ratio portfolio
    #1) Compute the local max SR portfolio between any two neighbor turning points
    w_sr,sr=[],[]
    for i in range(len(solution_set)-1):
        w0=np.copy(solution_set[i])
        w1=np.copy(solution_set[i+1])
        kargs={'minimum':False,'args':(w0,w1)}
        a,b=goldenSection(data_mean,data_covar,evalSR,0,1,**kargs)
        w_sr.append(a*w0+(1-a)*w1)
        sr.append(b)
    return max(sr),w_sr[sr.index(max(sr))]
#---------------------------------------------------------------
def evalSR(data_mean,data_covar, a,w0,w1):
    # Evaluate SR of the portfolio within the convex combination
    w=a*w0+(1-a)*w1
    b=np.dot(w.T,data_mean)[0,0]
    c=np.dot(np.dot(w.T,data_covar),w)[0,0]**.5
    return b/c
#---------------------------------------------------------------
def goldenSection(data_mean,data_covar,obj,a,b,**kargs):
    # Golden section method. Maximum if kargs['minimum']==False is passed 
    from math import log,ceil
    tol,sign,args=1.0e-9,1,None
    if 'minimum' in kargs and kargs['minimum']==False:sign=-1
    if 'args' in kargs:args=kargs['args']
    numIter=int(ceil(-2.078087*log(tol/abs(b-a))))
    r=0.618033989
    c=1.0-r
    # Initialize
    x1=r*a+c*b;x2=c*a+r*b
    f1=sign*obj(data_mean, data_covar, x1,*args);f2=sign*obj( data_mean, data_covar, x2,*args)
    # Loop
    for i in range(numIter):
        if f1>f2:
            a=x1
            x1=x2;f1=f2
            x2=c*a+r*b;f2=sign*obj(data_mean,data_covar, x2,*args)
        else:
            b=x2
            x2=x1;f2=f1
            x1=r*a+c*b;f1=sign*obj(data_mean,data_covar,x1,*args)
    if f1<f2:return x1,sign*f1
    else:return x2,sign*f2
#---------------------------------------------------------------
def efFrontier( data_mean, data_covar, solution_set, points):
    # Get the efficient frontier
    mu,sigma,weights=[],[],[]
    a=np.linspace(0,1,points/len(solution_set))[:-1] # remove the 1, to avoid duplications
    b=range(len(solution_set)-1)
    for i in b:
        w0,w1=solution_set[i],solution_set[i+1]
        if i==b[-1]:a=np.linspace(0,1,points/len(solution_set)) # include the 1 in the last iteration
        for j in a:
            w=w1*j+(1-j)*w0
            weights.append(np.copy(w))
            mu.append(np.dot(w.T,data_mean)[0,0])
            sigma.append(np.dot(np.dot(w.T,data_covar),w)[0,0]**.5)
    return mu,sigma,weights
#---------------------------------------------------------------    
def computeBi(c,bi):
    if c>0:
        bi=bi[1][0]
    if c<0:
        bi=bi[0][0]
    return bi
#---------------------------------------------------------------
def computeW(lambdas,covarF_inv,covarFB,meanF,wB):
    #1) compute gamma
    onesF=np.ones(meanF.shape)
    g1=np.dot(np.dot(onesF.T,covarF_inv),meanF)
    g2=np.dot(np.dot(onesF.T,covarF_inv),onesF)
    if wB==None:
        g,w1=float(-lambdas[-1]*g1/g2+1/g2),0
    else:
        onesB=np.ones(wB.shape)
        g3=np.dot(onesB.T,wB)
        g4=np.dot(covarF_inv,covarFB)
        w1=np.dot(g4,wB)
        g4=np.dot(onesF.T,w1)
        g=float(-lambdas[-1]*g1/g2+(1-g3+g4)/g2)
    #2) compute weights
    w2=np.dot(covarF_inv,onesF)
    w3=np.dot(covarF_inv,meanF)
    return -w1+g*w2+lambdas[-1]*w3,g
#---------------------------------------------------------------
def computeLambda(covarF_inv,covarFB,meanF,wB,i,bi):
    #1) C
    onesF=np.ones(meanF.shape)
    c1=np.dot(np.dot(onesF.T,covarF_inv),onesF)
    c2=np.dot(covarF_inv,meanF)
    c3=np.dot(np.dot(onesF.T,covarF_inv),meanF)
    c4=np.dot(covarF_inv,onesF)
    c=-c1*c2[i]+c3*c4[i]
    if c==0:return None,None
    #2) bi
    if type(bi)==list:bi=computeBi(c,bi)
    #3) Lambda
    if wB==None:
        # All free assets
        return float((c4[i]-c1*bi)/c),bi
    else:
        onesB=np.ones(wB.shape)
        l1=np.dot(onesB.T,wB)
        l2=np.dot(covarF_inv,covarFB)
        l3=np.dot(l2,wB)
        l2=np.dot(onesF.T,l3)
        return float(((1-l1+l2)*c4[i]-c1*(bi+l3[i]))/c),bi
#---------------------------------------------------------------
def getMatrices(data_mean, data_covar,solution_set,f):
    # Slice covarF,covarFB,covarB,meanF,meanB,wF,wB	
    covarF=reduceMatrix_gpu(data_covar,f,f)
    meanF=reduceMatrix_cpu(data_mean,f,[0])
    b=getB(data_mean, f)	
    covarFB=reduceMatrix_gpu(data_covar,f,b)
    wB=reduceMatrix_cpu(solution_set[-1],b,[0])
    return covarF,covarFB,meanF,wB
#---------------------------------------------------------------
def solve(path):
    #2) Load data, set seed
    headers=open(path,'r').readline()[:-1].split(',')
    data=np.genfromtxt(path,delimiter=',',skip_header=1) # load as numpy array
    data_mean=np.array(data[:1]).T
    lower_bound=np.array(data[1:2]).T
    upper_bound=np.array(data[2:3]).T
    data_covar=np.array(data[3:])
    lambdas=[] # lambdas
    gammas=[] # gammas
    free_weights=[] # free weights
    solution_set=[] # solution
#############CREATE GPU_ARRAYS for data read into from file #####################
  ###means for all data
#    device_mean = gpuarray.to_gpu(data_mean.astype(np.float32))
  ###covariance matrix for all data
#    device_covar =gpuarray.to_gpu(data_covar.astype(np.float32))
  ###upper bounds for data
#    device_upper_bound = gpuarray.to_gpu(upper_bound.astype(np.float32))
  ###lower bounds for data
#    device_lower_bound = gpuarray.to_gpu(lower_bound.astype(np.float32))
#    print "gpu_mean \n {0} \n gpu_covar \n {1} \n gpu_uB \n{2} \n gpu_lB \n{3} \n".format(device_mean, device_covar, \
#                                                                                      device_upper_bound, device_lower_bound)
    # Compute the turning points,free sets and weights
    f,w=initAlgo(data_mean,lower_bound, upper_bound)
    solution_set.append(np.copy(w)) # store solution
    lambdas.append(None)
    gammas.append(None)
    free_weights.append(f[:])
    while True:
        #1) case a): Bound one free weight
        l_in=None
        if len(f)>1:
            covarF,covarFB,meanF,wB=getMatrices(data_mean, data_covar,solution_set,f)
            covarF_inv=np.linalg.inv(covarF)
            j=0
            for i in f:
                l,bi=computeLambda(covarF_inv,covarFB,meanF,wB,j,[lower_bound[i],upper_bound[i]])
                if l>l_in:l_in,i_in,bi_in=l,i,bi
                j+=1
        #2) case b): Free one bounded weight
        l_out=None
        if len(f)<data_mean.shape[0]:
            b=getB(data_mean, f)
            for i in b:
                covarF,covarFB,meanF,wB=getMatrices(data_mean, data_covar,solution_set,f+[i])
                covarF_inv=np.linalg.inv(covarF)
                l,bi=computeLambda(covarF_inv,covarFB,meanF,wB,meanF.shape[0]-1,solution_set[-1][i])
                if (lambdas[-1]==None or l<lambdas[-1]) and l>l_out:l_out,i_out=l,i                
        if (l_in==None or l_in<0) and (l_out==None or l_out<0):
            #3) compute minimum variance solution
            lambdas.append(0)
            covarF,covarFB,meanF,wB=getMatrices(data_mean, data_covar,solution_set,f)
            covarF_inv=np.linalg.inv(covarF)
            meanF=np.zeros(meanF.shape)
        else:
            #4) decide lambda
            if l_in>l_out:
                lambdas.append(l_in)
                f.remove(i_in)
                w[i_in]=bi_in # set value at the correct boundary
            else:
                lambdas.append(l_out)
                f.append(i_out)
            covarF,covarFB,meanF,wB=getMatrices(data_mean, data_covar,solution_set,f)
            covarF_inv=np.linalg.inv(covarF)
        #5) compute solution vector
        wF,g=computeW( lambdas,covarF_inv,covarFB,meanF,wB)
        for i in range(len(f)):w[f[i]]=wF[i]
        solution_set.append(np.copy(w)) # store solution
        gammas.append(g)
        free_weights.append(f[:])
        if lambdas[-1]==0:break
    #6) Purge turning points
    purgeNumErr(free_weights ,gammas ,lambdas,lower_bound,upper_bound,solution_set,10e-10)
    purgeExcess(data_mean, lambdas, gammas, free_weights, solution_set)
    return data_mean, data_covar, solution_set

#---------------------------------------------------------------
def plot2D(x,y,xLabel='',yLabel='',title='',pathChart=None):
    import matplotlib.pyplot as mpl
    fig=mpl.figure()
    ax=fig.add_subplot(1,1,1) #one row, one column, first plot
    ax.plot(x,y,color='blue')
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel,rotation=90)
    mpl.xticks(rotation='vertical')
    mpl.title(title)
    if pathChart==None:
        mpl.show()
    else:
        mpl.savefig(pathChart)
    mpl.clf() # reset pylab
    return
#---------------------------------------------------------------
def main():

    #1) Path
    orig = "./CLA_Data.csv"
    n_86 = './86/86'
    n_118 ='./118/118'
    n_139 ='./139/139'
    n_160 ='./160/160'
    n_174 ='./174/174'
    n_257 ='./257/257'
    n_310 ='./310/310'
    n_346 ='./346/346'
    n_390 ='./390/390'
    n_463 ='./463/463'
    n_501 ='./501/501'
    
    def f(x):
        return {
            0:  orig,
            1:  n_86,
            2:  n_118,
            3:  n_139,
            4:  n_160,
            5:  n_174,
            6:  n_257,
            7:  n_310,
            8:  n_346,
            9:  n_390,
            10: n_463,
            11: n_501,
        }[x]
    
    for i in xrange(0,1):
    #    %%time
        path= f(i)
        #start the profile information
        pr = cProfile.Profile()
        pr.enable()
#         print 'The values in the mean array: \n {} '.format( mean)	
#        start = time.time()
        mean, covar, sol =  solve(path)
#        end = time.time()
        #4) Plot frontier
        mu,sigma,weights=efFrontier(mean, covar, sol,100)
#        plot2D(sigma,mu,'Risk','Expected Excess Return','CLA-derived Efficient Frontier')
        #5) Get Maximum Sharpe ratio portfolio
        sr,w_sr=getMaxSR(mean, covar, sol)
        dont_know_what_this_is = (np.dot(np.dot(w_sr.T, covar),w_sr)[0,0]**.5,sr)
#        print type(dont_know_what_this_is)
#        print 'The max sharpe ratio is \n {}'.format(dont_know_what_this_is)
#        print'The weights for the max sharpe ratio portfolio are \n {}'.format( w_sr)
        #6) Get Minimum Variance portfolio
        mv,w_mv=getMinVar(covar, sol)
#        print"Run for {} file".format(path)
#        print"The min variance is \n {} ".format( mv)
#        print "The weights for min variance are: \n {} ".format(w_mv)
#        print "\n \n"
        print "#############################START OF PROFILE INFORMATION ######################"
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()
#        print sol # print all turning points
    return
#---------------------------------------------------------------
# Boilerplate
if __name__=='__main__':main()

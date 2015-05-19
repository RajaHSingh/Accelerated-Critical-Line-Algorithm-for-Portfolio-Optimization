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
    import numpy as np
    import CLA
    #1) Path
    path='./CLA_Data.csv'
    #2) Load data, set seed
    headers=open(path,'r').readline()[:-1].split(',')
    data=np.genfromtxt(path,delimiter=',',skip_header=1) # load as numpy array
    mean=np.array(data[:1]).T
    lB=np.array(data[1:2]).T
    uB=np.array(data[2:3]).T
    covar=np.array(data[3:])
    #3) Invoke object
    cla=CLA.CLA(mean,covar,lB,uB)
    cla.solve()
    #4) Plot frontier
    mu,sigma,weights=cla.efFrontier(100)
    plot2D(sigma,mu,'Risk','Expected Excess Return','CLA-derived Efficient Frontier')
    #5) Get Maximum Sharpe ratio portfolio
    sr,w_sr=cla.getMaxSR()
    y = (np.dot(np.dot(w_sr.T,cla.covar),w_sr)[0,0]**.5,sr)
    print 'The max sharpe ratio is \n {}'.format(y)
    print'The weights for the max sharpe ratio portfolio are \n {}'.format( w_sr)
    #6) Get Minimum Variance portfolio
    mv,w_mv=cla.getMinVar()
    print"The min variance is \n {} ".format( mv)
    print "The weights for min variance are: \n {} ".format(w_mv)
    print cla.w # print all turning points
    return
#---------------------------------------------------------------
# Boilerplate
if __name__=='__main__':main()


class stats:

    def q_test(suspect, data):
        gap = 0
        try:
            if len(data) <=2:
                return
            else:
                #insertion sort alg
                for i in range(len(data)):
                    cursor = data[i]
                    pos = i
                    while pos > 0 and data[pos - 1] > cursor:
                        # Swap the number down the list
                        data[pos] = data[pos - 1]
                        pos = pos - 1
                    # Break and do the final swap
                    data[pos] = cursor
                #get the gap
                for i in range(len(data)):
                    # if the suspect is at the end of the dataset
                    if  data[len(data)-1] == suspect:
                        gap = suspect - data[i-1]
                    elif data[0] == suspect:
                        gap = suspect - data[i+1]
                if gap < 0:
                    gap = -1*gap
                return gap/(data[len(data)-1] - data[0])
        except:
            return 0

    def std(data):
        try:
            if len(data) <= 2:
                return 0
            else:
                for i in range(len(data)):
                    xbar += data[i]
                xbar = xbar/len(data)
                for i in range(len(data)):
                    sig += ((data[i] - xbar)/len(data))**2
                sig = sum(sig)
                return sig
        except:
            return 0

    def mean(data):
        for i in range(len(data)):
            xbar += data[i]
        xbar = xbar/len(data)
        return xbar

    def chi_sq_v(fit, data):
        x = 0
        if len(fit) != len(data):
            return 0
        else:
            for i in range(len(data)):
                x += (fit[i] - data[i])**2/data[i]
            s = x/(len(data)-1)
            return s

    def lin_reg(xdata,ydata):
        try:
            n = len(xdata)
            Sx = sum(xdata)
            Sy = sum(ydata)
            sxx = []
            syy = []
            sxy = []
            for i in range(n):
                sxx.append((xdata[i])**2)
                syy.append((ydata[i])**2)
                sxy.append(xdata[i]*ydata[i])
            Sxx = sum(sxx)
            Syy = sum(syy)
            Sxy = sum(sxy)
            xbar = Sx/n
            ybar = Sy/n
            xybar = Sxy/n
            xsqbar = Sxx/n
            ysqbar = Syy/n
            a = (n*Sxy - Sx*Sy)/(n*Sxx - Sx**2)
            b = 1/n*(Sy - a*Sx)
            r2 = (xybar - xbar*ybar)**2/((xsqbar - xbar**2)*(ysqbar - ybar**2))
            return a,b,r2
        except:
            if len(xdata) != len(ydata):
                print("Lengths don't match: x -> %i, y -> %i" % (len(xdata), len(ydata)))
            elif len(xdata) or len(ydata) < 2:
                print("Dataset too small")
            else:
                print("Invalid Data type!")

    def mse(yfit,ydata):
        try:
            if len(yfit) != len(ydata):
                print("Lengths don't match: yfit -> %i, ydata -> %i" % (len(yfit), len(ydata)))
            else:
                n = len(ydata)
                dy = 0
                for i in range(n):
                    dy += (ydata[i] - yfit[i])**2
                return dy/n
        except:
            pass

    def curve_fit(function,x,y,p0):
        try:
            from scipy.optimize import curve_fit
        except:
            print("Couldn't find Scipy.")
        try:
            coef, err_raw = curve_fit(function,x,y,p0)
            err = []
            for i in range(len(coef)):
                err.append((err_raw[i][i])**(1/2))
            return coef,err
        except:
            if len(x) != len(y):
                print("Lengths don't match: x -> %i, y -> %i" % (len(x), len(y)))
            elif len(x) or len(y) < 2:
                print("Dataset too small")
            else:
                print("Invalid Data type!")

    def help(function=None):
        if function == None:
            print("The functions available are:")
            lst = [(name) for name, t in test_class.__dict__.items() if type(t).__name__ == 'function' and not name.startswith('__')]
            for i in lst:
                print(i)
        else:
            params = {"q_test":stats.q_test.__code__.co_varnames[:stats.q_test.__code__.co_argcount],
            "std":stats.std.__code__.co_varnames[:stats.std.__code__.co_argcount],
            "mean":stats.mean.__code__.co_varnames[:stats.mean.__code__.co_argcount],
            "chi_sq_v":stats.chi_sq_v.__code__.co_varnames[:stats.chi_sq_v.__code__.co_argcount],
            "lin_reg":stats.lin_reg.__code__.co_varnames[:stats.lin_reg.__code__.co_argcount],
            "mse":stats.mse.__code__.co_varnames[:stats.mse.__code__.co_argcount],
            "curve_fit":stats.curve_fit.__code__.co_varnames[:stats.curve_fit.__code__.co_argcount]}
            print(params[function])

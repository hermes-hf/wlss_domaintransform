import numpy as np
import source.vis as vis

def applyDomainTransformFilter_yaxis(a, dt, input_img):
    fwd = input_img*0
    bwd = input_img*0
    N,M = np.shape(input_img)

    #fwd pass
    fwd[:,0] = a[:,0]*input_img[:,0]
    for y in range(1,M):
        fwd[:,y] = a[:,y]*input_img[:,y] + np.exp(-abs(dt[:,y]))*fwd[:,y-1]

    #bwd pass
    bwd[:,M-1] = a[:,M-1]*input_img[:,M-1]
    for y in range(M-2,-1,-1):
        bwd[:,y] = a[:,y+1]*input_img[:,y+1]*np.exp(-abs(dt[:,y+1])) + np.exp(-abs(dt[:,y+1]))*bwd[:,y+1]

    return a*(fwd + bwd)

def applyDomainTransformFilter_xaxis(a, dt, input_img):
    img = applyDomainTransformFilter_yaxis(np.transpose(a), 
                                           np.transpose(dt), np.transpose(input_img))
    return np.transpose(img)

def getDomainTransformCoefficients(luminance, sigma_s, sigma_r, alpha_r, tol = 1e-3):
    N,M = np.shape(luminance)
    ones_mat = np.ones((N,M))
    success = False

    sigma_h = np.sqrt(2)
    dt_x = luminance*0
    dt_y = luminance*0

    dimg_dx = luminance*0
    dimg_dx[1:,:] = luminance[1:,:] - luminance[0:-1,:]

    dimg_dy = luminance*0
    dimg_dy[:,1:] = luminance[:,1:] - luminance[:,0:-1]

    dt_x = np.sqrt((sigma_h/sigma_s)**2 + (sigma_h/sigma_r * abs(dimg_dx)**alpha_r)**2 )
    dt_y = np.sqrt((sigma_h/sigma_s)**2 + (sigma_h/sigma_r * abs(dimg_dy)**alpha_r)**2 )
    dt_x[0,:] = 0
    dt_y[:,0] = 0

    ax = luminance*0 + 1
    ay = luminance*0 + 1

    it = 0
    err = 1
    max_it = 1000
    while err>tol and it < max_it:
        it += 1
        filtered_y = applyDomainTransformFilter_yaxis(ay, dt_y, ones_mat)
        filtered_x = applyDomainTransformFilter_xaxis(ax, dt_x, ones_mat)
        #check tolerance
        if it%5==0:
            err_y = np.linalg.norm(filtered_y*ay-ones_mat)/np.linalg.norm(ones_mat)
            err_x = np.linalg.norm(filtered_x*ax-ones_mat)/np.linalg.norm(ones_mat)
            err = max(err_y,err_x)
        ay = (ay + 1/filtered_y)/2.0
        ax = (ax + 1/filtered_x)/2.0
    if err>tol:
        return ax, ay, dt_x, dt_y, Wx, Wy, success
    success = True
        
            

    Wx = ax*0
    Wy = ay*0

    Wx[1:,:] = np.exp(-abs(dt_x[1:,:]))/(1- np.exp(-2*abs(dt_x[1:,:])))/(ax[1:,:]*ax[0:-1,:])
    Wy[:,1:] = np.exp(-abs(dt_y[:,1:]))/(1- np.exp(-2*abs(dt_y[:,1:])))/(ay[:,1:]*ay[:,0:-1])

    
    return ax, ay, dt_x, dt_y, Wx, Wy, success
    
def get1D_DomainTransformCoefficients(N, dt_y, tol = 1e-3):
    ones_mat = np.ones((1,N))
    ay = ones_mat*1

    it = 0
    err = 1
    max_it = 1000
    while err>tol and it < max_it:
        it += 1
        filtered_y = applyDomainTransformFilter_yaxis(ay, dt_y, ones_mat)
        #check tolerance
        if it%5==0:
            err = np.linalg.norm(filtered_y*ay-ones_mat)/np.linalg.norm(ones_mat)
        ay = (ay + 1/filtered_y)/2.0
        
    Wy = ay*0

    Wy[:,1:] = np.exp(-abs(dt_y[:,1:]))/(1- np.exp(-2*abs(dt_y[:,1:])))/(ay[:,1:]*ay[:,0:-1])

    
    return Wy
    

def tikhonov1D_y(input_img, W):
    M,N = np.shape(W)
    A = W*0
    B = W*0
    C = W*1

    #compute B
    B[:,N-1] = 1 + W[:,N-1]
    for y in range(N-2,-1,-1):
        B[:,y] = 1 + W[:,y] + W[:,y+1] - W[:,y+1]**2/(B[:,y+1])

    #compute A
    A[:,0:-1] = W[:,1:]/B[:,1:]
    A[:,-1] = 0

    #bwd filter
    bwd = input_img*1
    for y in range(N-2,-1,-1):
        bwd[:,y] = A[:,y]*bwd[:,y+1] + input_img[:,y]
    
    fwd = bwd/B
    for y in range(1,N):
        fwd[:,y] = (bwd[:,y] + C[:,y]*fwd[:,y-1])/B[:,y]
    
    return fwd

def tikhonov1D_x(input_img, W):
    img = np.transpose(input_img)
    Wx = np.transpose(W)
    res = tikhonov1D_y(img, Wx)
    return np.transpose(res)


def admm_method_gastal(input_img, sigma_s, sigma_r, alpha_r, tol=1e-3, rho=10, bound = 0, channels = 1):
  
    luminance = vis.get_luminance(input_img)
    mono_image = luminance*0
    ax, ay, dt_x, dt_y, Wx, Wy, success = getDomainTransformCoefficients(luminance, sigma_s, sigma_r, alpha_r, tol = tol)
    max_it = 10000
    output_image = input_img*0
    output_data = dict()

    if success == False:
        output_data["success"] = False
        return output_image, output_data #failed in acquiring coefficients

    for c in range(channels):
        if channels==1:
            mono_image = luminance*1
        else:
            mono_image = input_img[:,:,c]*1

        X = mono_image*1
        Y = mono_image*1
        U = X*0
        it = 0
        err = 1.0
        while (err > tol) and (it < max_it):
            X = tikhonov1D_x((mono_image + rho*Y - U)/(1+rho), 2*Wx/(1+rho))
            Y = tikhonov1D_y((mono_image + rho*X + U)/(1+rho), 2*Wy/(1+rho))
            U = U + rho*(X-Y)
            it += 1
            #check error each 10 iterations
        
            if it % 10 == 0:
                err = np.linalg.norm(X - Y)/np.linalg.norm(mono_image)
        output_image[:,:,c] = X*1

    if channels==1:
        for c in range(3):
            output_image[:,:,c] = X*1

    if it >= max_it:
        output_data["success"] = False
    else:
        output_data["success"] = True
    output_data["admm_it"] = it
    
    return output_image, output_data
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg
import source.domain_transform_filter as dtf
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import source.vis as vis
from scipy.sparse.linalg import lsqr

def get_wx_wy(luminance,  lambda_val, alpha, epsilon, tol = 1e-3):
    N,M = np.shape(luminance)
    lx = lambda_val
    ly = lambda_val
    dimg_dx = luminance*0
    dimg_dx[1:,:] = luminance[1:,:] - luminance[0:-1,:]

    dimg_dy = luminance*0
    dimg_dy[:,1:] = luminance[:,1:] - luminance[:,0:-1]

  
    Wx = lx/(abs(dimg_dx)**alpha + epsilon)
    Wy = ly/(abs(dimg_dy)**alpha + epsilon)

    Wx[0,:] = 0
    Wy[:,0] = 0

    return Wx, Wy

def farbman_wlss_sparse(img, Wx, Wy):
    n,m = np.shape(img)    

    def wx(x,y):
        if x>=n or x<0 or y>=m or y<0:
            return 0 
        return Wx[x,y]
    def wy(x,y):
        if x>=n or x<0 or y>=m or y<0:
            return 0 
        return Wy[x,y]
    
    I = np.zeros(5*n*m)
    J = np.zeros(5*n*m)
    V = np.zeros(5*n*m)
    
    #fill matrix
    
    nrow = 0
    for x in range(n):
        for y in range(m):
            i = x*n + y
            I[nrow] = i
            J[nrow] = i
            V[nrow] = 1 + wx(x,y) + wx(x+1,y) + wy(x,y) + wy(x,y+1)
            nrow +=1
      
    for x in range(n):
        for y in range(m):
            i = (x)*n + y
            j = i + 1
            if not (i>=n*m or i<0 or j>= n*m or j<0):
                if not (x>=n or x<0 or y+1>=m or y+1<0):
                    I[nrow] = i
                    J[nrow] = j
                    V[nrow] = -wy(x,y+1)
                    nrow +=1
    
      
    for x in range(n):
        for y in range(m):
            i = (x)*n + y
            j = i - 1
            if not(i>=n*m or i<0 or j>= n*m or j<0):
                if not(x>=n or x<0 or y>=m or y<0):
                    I[nrow] = i
                    J[nrow] = j
                    V[nrow] = -wy(x,y)
                    nrow +=1
            
    for x in range(n):
        for y in range(m):
            i = (x)*n + y
            j = i + m
            if not(i>=n*m or i<0 or j>= n*m or j<0):
                if not(x+1>=n or x+1<0 or y>=m or y<0):
                    I[nrow] = i
                    J[nrow] = j
                    V[nrow] = -wx(x+1,y)
                    nrow +=1
        
    for x in range(n):
        for y in range(m):
            i = (x)*n + y
            j = i - m
            if not(i>=n*m or i<0 or j>= n*m or j<0):
                if not(x-1>=n or x-1<0 or y>=m or y<0):
                    I[nrow] = i
                    J[nrow] = j
                    V[nrow] = -wx(x,y)
                    nrow +=1

   
    
    I = I[0:nrow]
    J = J[0:nrow]
    V = V[0:nrow]
    A = csr_matrix((V, (I, J)), shape=(n*m, n*m))

    
    return A


def farbman_wlss_sparse_x(img, Wx):
    n,m = np.shape(img)    

    def wx(x,y):
        if x>=n or x<0 or y>=m or y<0:
            return 0 
        return Wx[x,y]
   
    
    I = np.zeros(5*n*m)
    J = np.zeros(5*n*m)
    V = np.zeros(5*n*m)
    
    #fill matrix
    
    nrow = 0
    for x in range(n):
        for y in range(m):
            i = x*n + y
            I[nrow] = i
            J[nrow] = i
            V[nrow] = 1 + wx(x,y) + wx(x+1,y)
            nrow +=1

            
    for x in range(n):
        for y in range(m):
            i = (x)*n + y
            j = i + m
            if not(i>=n*m or i<0 or j>= n*m or j<0):
                if not(x+1>=n or x+1<0 or y>=m or y<0):
                    I[nrow] = i
                    J[nrow] = j
                    V[nrow] = -wx(x+1,y)
                    nrow +=1
        
    for x in range(n):
        for y in range(m):
            i = (x)*n + y
            j = i - m
            if not(i>=n*m or i<0 or j>= n*m or j<0):
                if not(x-1>=n or x-1<0 or y>=m or y<0):
                    I[nrow] = i
                    J[nrow] = j
                    V[nrow] = -wx(x,y)
                    nrow +=1

    I = I[0:nrow]
    J = J[0:nrow]
    V = V[0:nrow]
    A = csr_matrix((V, (I, J)), shape=(n*m, n*m))

    
    return A


def farbman_wlss_sparse_y(img, Wy):
    n,m = np.shape(img)    

    def wy(x,y):
        if x>=n or x<0 or y>=m or y<0:
            return 0 
        return Wy[x,y]
    
    I = np.zeros(5*n*m)
    J = np.zeros(5*n*m)
    V = np.zeros(5*n*m)
    
    #fill matrix
    
    nrow = 0
    for x in range(n):
        for y in range(m):
            i = x*n + y
            I[nrow] = i
            J[nrow] = i
            V[nrow] = 1 + wy(x,y) + wy(x,y+1)
            nrow +=1
      
    for x in range(n):
        for y in range(m):
            i = (x)*n + y
            j = i + 1
            if not (i>=n*m or i<0 or j>= n*m or j<0):
                if not (x>=n or x<0 or y+1>=m or y+1<0):
                    I[nrow] = i
                    J[nrow] = j
                    V[nrow] = -wy(x,y+1)
                    nrow +=1
    
      
    for x in range(n):
        for y in range(m):
            i = (x)*n + y
            j = i - 1
            if not(i>=n*m or i<0 or j>= n*m or j<0):
                if not(x>=n or x<0 or y>=m or y<0):
                    I[nrow] = i
                    J[nrow] = j
                    V[nrow] = -wy(x,y)
                    nrow +=1
            

    I = I[0:nrow]
    J = J[0:nrow]
    V = V[0:nrow]
    A = csr_matrix((V, (I, J)), shape=(n*m, n*m))

    
    return A




def solve_hermitian_system(A, g, tol=1e-10, max_iter=1000):

    """
    Solves the system A.f = g where A is a Hermitian matrix using the Conjugate Gradient method.
    
    Parameters:
    A (ndarray or sparse matrix): Hermitian matrix (n x n)
    g (ndarray): Input vector (n)
    tol (float): Tolerance for convergence (default is 1e-10)
    max_iter (int): Maximum number of iterations (default is 1000)
    
    Returns:
    f (ndarray): Solution vector (n)
    info (int): Information flag (0 if successful, >0 if max iterations reached)
    """
    # Use the Conjugate Gradient method to solve the system
    f, info = cg(A, np.matrix.flatten(g), rtol=tol, maxiter=max_iter, x0 = np.matrix.flatten(g))
    
    #if info == 0:
    #    print("Converged successfully!")
    #elif info > 0:
    #    print(f"Conjugate Gradient did not converge within {max_iter} iterations.")
   # else:
    #    print("Conjugate Gradient failed to converge due to an error.")
    
    return f, info

def apply_farbman_method(img, lambda_val, alpha, epsilon, tol = 1e-7, max_iter = 100, bound = 0):
    Wx, Wy=get_wx_wy(img, lambda_val, alpha, epsilon, bound = bound)
    A = farbman_wlss_sparse(img, Wx,Wy)
    output, info = solve_hermitian_system(A, img, tol=tol, max_iter=max_iter)
    output = np.reshape(output, np.shape(img))
    return output


def apply_farbman_method_x(img, Wx,tol = 1e-7, max_iter = 100):
    A = farbman_wlss_sparse_x(img, Wx)
    output, info = solve_hermitian_system(A, img, tol=tol, max_iter=max_iter)
    output = np.reshape(output, np.shape(img))
    return output

def apply_farbman_method_y(img, Wy, tol = 1e-7, max_iter = 100):
    A = farbman_wlss_sparse_y(img, Wy)
    output, info = solve_hermitian_system(A, img, tol=tol, max_iter=max_iter)
    output = np.reshape(output, np.shape(img))
    return output


def admm_method_farbman(input_img, sigma_s, sigma_r, alpha_r, tol=1e-3, rho=10, bound = 0, channels = 1):
    #apply conversion
    lambda_val = sigma_r**2/2
    epsilon = (sigma_r/sigma_s)**2
    alpha = 2*alpha_r

    luminance = vis.get_luminance(input_img)
    mono_image = luminance*0
    Wx, Wy = get_wx_wy(luminance, lambda_val, alpha, epsilon)
    max_it = 10000
    output_image = input_img*0
    output_data = dict()

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
            X = dtf.tikhonov1D_x((mono_image + rho*Y - U)/(1+rho), 2*Wx/(1+rho))
            Y = dtf.tikhonov1D_y((mono_image + rho*X + U)/(1+rho), 2*Wy/(1+rho))
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


    
def lsqr_method_farbman(input_img, sigma_s, sigma_r, alpha_r, tol=1e-3, rho=10, bound = 0, channels = 1):
    #apply conversion
    lambda_val = sigma_r**2/2
    epsilon = (sigma_r/sigma_s)**2
    alpha = 2*alpha_r

    luminance = vis.get_luminance(input_img)
    mono_image = luminance*0
    Wx, Wy = get_wx_wy(luminance, lambda_val, alpha, epsilon)
    A = farbman_wlss_sparse(luminance, Wx, Wy)
    output_image = input_img*0
    output_data = dict()

    for c in range(channels):
        if channels==1:
            mono_image = luminance*1
        else:
            mono_image = input_img[:,:,c]*1

        sol = lsqr(A, np.ndarray.flatten(mono_image))

        output_image[:,:,c] = np.reshape(sol,np.shape(mono_image))

    if channels==1:
        for c in range(3):
            output_image[:,:,c] = X*1

    if it >= max_it:
        output_data["success"] = False
    else:
        output_data["success"] = True
    output_data["admm_it"] = it
    
    return output_image, output_data
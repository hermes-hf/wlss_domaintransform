import numpy as np
import source.vis as vis


def multiply2by2(A11, A12, A21, A22, B11, B12, B21, B22):
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22

    return C11, C12, C21, C22


def normalize2by2(A11, A12, A21, A22):
    C = np.maximum(A11, A12)
    C = np.maximum(C, A21)
    C = np.maximum(C, A22)
    return A11/C, A12/C, A21/C, A22/C


def makeA21positive(A11, A12, A21, A22):
    return A11*A21, A12*A21, A21*A21, A22*A21


def inverse2by2(A11, A12, A21, A22):
    det = A11*A22 - A12*A21
    return A22/det, -A12/det, -A21/det, A11/det


def quadraticSolve(A11, A12, A21, A22):
    # solution to B(x) = (A11 B(x) + A12)/(A21 B(x) + A22)
    return (A11-A22 + np.sqrt(A11**2 + 4*A12*A21 - 2*A11*A22+A22**2))/(2*A21)


def linearSolveBwd(A11, A12, A21, A22):
    # solution to bwd(x) = (A11 bwd(x) + A12)/A22
    return A12/(A22-A11)


def compute_circularFwd_dtf(input_img, a, dt):
    M, N = np.shape(input_img)
    fwd = input_img*0

    A11 = a[:, 0]*0 + 1
    A12 = 0*A11
    A21 = 0*A11
    A22 = 1*A11

    for y in range(N):
        B11 = np.exp(abs(dt[:, (y+1) % N]))
        B12 = -input_img[:, (y+1) % N] * \
            np.exp(abs(dt[:, (y+1) % N]))*a[:, (y+1) % N]
        B21 = A11*0
        B22 = 1

        A11, A12, A21, A22 = multiply2by2(
            A11, A12, A21, A22, B11, B12, B21, B22)
        A11, A12, A21, A22 = normalize2by2(A11, A12, A21, A22)
    # solve linear equation for fwd[0]
    fwd[:, 0] = linearSolveBwd(A11, A12, A21, A22)

    # compute other poitns from fwd[0]
    for y in range(1, N):
        fwd[:, y] = a[:, y]*input_img[:, y] + \
            np.exp(-abs(dt[:, y]))*fwd[:, y-1]

    return fwd


def compute_circularBwd_dtf(input_img, a, dt):
    M, N = np.shape(input_img)
    bwd = input_img*0

    A11 = a[:, 0]*0 + 1
    A12 = 0*A11
    A21 = 0*A11
    A22 = 1*A11

    for k in range(N):
        y = (0-k) % N
        B11 = np.exp(abs(dt[:, y % N]))
        B12 = -input_img[:, y % N]*a[:, y % N]
        B21 = A11*0
        B22 = 1

        A11, A12, A21, A22 = multiply2by2(
            A11, A12, A21, A22, B11, B12, B21, B22)
        A11, A12, A21, A22 = normalize2by2(A11, A12, A21, A22)
    # solve linear equation for bwd[0]
    bwd[:, 0] = linearSolveBwd(A11, A12, A21, A22)

    # compute other poitns from bwd[0]
    # bwd pass
    bwd[:, N-1] = a[:, 0]*input_img[:, 0] * \
        np.exp(-abs(dt[:, 0])) + np.exp(-abs(dt[:, 0]))*bwd[:, 0]
    for y in range(N-2, 0, -1):
        bwd[:, y] = a[:, y+1]*input_img[:, y+1] * \
            np.exp(-abs(dt[:, y+1])) + np.exp(-abs(dt[:, y+1]))*bwd[:, y+1]
    return bwd


def applyDomainTransformFilter_yaxis(a, dt, input_img, normalize=True, mode='truncate'):
    if mode == 'truncate':
        fwd = input_img*0
        bwd = input_img*0
        M, N = np.shape(input_img)

        # fwd pass
        fwd[:, 0] = a[:, 0]*input_img[:, 0]
        for y in range(1, N):
            fwd[:, y] = a[:, y]*input_img[:, y] + \
                np.exp(-abs(dt[:, y]))*fwd[:, y-1]

        # bwd pass
        for y in range(N-2, -1, -1):
            bwd[:, y] = a[:, y+1]*input_img[:, y+1] * \
                np.exp(-abs(dt[:, y+1])) + np.exp(-abs(dt[:, y+1]))*bwd[:, y+1]
    elif mode == 'circular':
        fwd = compute_circularFwd_dtf(input_img, a, dt)
        bwd = compute_circularBwd_dtf(input_img, a, dt)
    if normalize:
        return a*(fwd + bwd)
    else:
        return (fwd + bwd)


def applyDomainTransformFilter_xaxis(a, dt, input_img, normalize=True, mode='truncate'):
    img = applyDomainTransformFilter_yaxis(np.transpose(a),
                                           np.transpose(dt), np.transpose(input_img), normalize, mode=mode)
    return np.transpose(img)


def getDomainTransformCoefficients(luminance, sigma_s, sigma_r, alpha_r, tol=1e-3, mode='truncate'):
    N, M = np.shape(luminance)
    ones_mat = np.ones((N, M))
    success = False

    sigma_h = np.sqrt(2)

    if mode == 'truncate':
        dimg_dx = luminance*0
        dimg_dx[1:, :] = luminance[1:, :] - luminance[0:-1, :]

        dimg_dy = luminance*0
        dimg_dy[:, 1:] = luminance[:, 1:] - luminance[:, 0:-1]

        dt_x = np.sqrt((sigma_h/sigma_s)**2 +
                       (sigma_h/sigma_r * abs(dimg_dx)**alpha_r)**2)
        dt_y = np.sqrt((sigma_h/sigma_s)**2 +
                       (sigma_h/sigma_r * abs(dimg_dy)**alpha_r)**2)

        dt_x[0, :] = 1e10
        dt_y[:, 0] = 1e10
    elif mode == 'circular':
        dimg_dx = luminance - np.roll(luminance, 1, axis=0)
        dimg_dy = luminance - np.roll(luminance, 1, axis=1)
        dt_x = np.sqrt((sigma_h/sigma_s)**2 +
                       (sigma_h/sigma_r * abs(dimg_dx)**alpha_r)**2)
        dt_y = np.sqrt((sigma_h/sigma_s)**2 +
                       (sigma_h/sigma_r * abs(dimg_dy)**alpha_r)**2)

    ax = luminance*0 + 1
    ay = luminance*0 + 1

    it = 0
    err = 1
    max_it = 1000
    while err > tol and it < max_it:
        it += 1
        filtered_y = applyDomainTransformFilter_yaxis(
            ay, dt_y, ones_mat, normalize=False, mode=mode)
        filtered_x = applyDomainTransformFilter_xaxis(
            ax, dt_x, ones_mat, normalize=False, mode=mode)
        # check tolerance
        if it % 5 == 0:
            err_y = np.linalg.norm(filtered_y*ay-ones_mat) / \
                np.linalg.norm(ones_mat)
            err_x = np.linalg.norm(filtered_x*ax-ones_mat) / \
                np.linalg.norm(ones_mat)
            err = max(err_y, err_x)
        ay = (ay + 1/filtered_y)/2.0
        ax = (ax + 1/filtered_x)/2.0
    if err > tol:
        success = False
    else:
        success = True

    if mode == 'truncate':
        Wx = ax*0
        Wy = ay*0

        Wx[1:, :] = np.exp(-abs(dt_x[1:, :])) / \
            (1 - np.exp(-2*abs(dt_x[1:, :])))/(ax[1:, :]*ax[0:-1, :])
        Wy[:, 1:] = np.exp(-abs(dt_y[:, 1:])) / \
            (1 - np.exp(-2*abs(dt_y[:, 1:])))/(ay[:, 1:]*ay[:, 0:-1])

    elif mode == 'circular':
        Wx = np.exp(-abs(dt_x)) / \
            (1 - np.exp(-2*abs(dt_x)))/(ax * np.roll(ax, 1, axis=0))
        Wy = np.exp(-abs(dt_y)) / \
            (1 - np.exp(-2*abs(dt_y)))/(ay * np.roll(ay, 1, axis=0))

    return ax, ay, dt_x, dt_y, Wx, Wy, success


def compute_circularB(W):
    # compute B[0] using fwd recursion B[x] from B[x-1], it is easier to solve than the usual bwd recursion
    M, N = np.shape(W)
    B = W*0

    # compute B[0]: must apply matrices from 0 to N-1
    # buffer matrices for computing B
    # start with identity matrix
    A11 = W[:, 0]*0 + 1
    A12 = 0*A11
    A21 = 0*A11
    A22 = 1*A11

    for k in range(N):
        y = (0-k) % N
        B11 = A11*0
        B12 = W[:, y]**2
        B21 = A11*0 - 1
        B22 = 1 + W[:, (y-1) % N] + W[:, y]

        A11, A12, A21, A22 = multiply2by2(
            A11, A12, A21, A22, B11, B12, B21, B22)
        A11, A12, A21, A22 = makeA21positive(A11, A12, A21, A22)
        A11, A12, A21, A22 = normalize2by2(A11, A12, A21, A22)

    # solve quadratic equation for B[0]
    B[:, 0] = quadraticSolve(A11, A12, A21, A22)

    # compute other points from B0:
    B[:, N-1] = 1 + W[:, N-1] + W[:, 0] - W[:, 0]**2/B[:, 0]

    for y in range(N-2, 0, -1):
        B[:, y] = 1 + W[:, y] + W[:, y+1] - W[:, y+1]**2/B[:, y+1]

    return B


def compute_circularBwd(input_img, A):
    M, N = np.shape(input_img)
    bwd = input_img*0

    A11 = A[:, 0]*0 + 1
    A12 = 0*A11
    A21 = 0*A11
    A22 = 1*A11

    for k in range(1, N+1):
        y = (0-k) % N
        B11 = A11*0 + 1
        B12 = -input_img[:, y]
        B21 = A11*0
        B22 = A[:, y]

        A11, A12, A21, A22 = multiply2by2(
            A11, A12, A21, A22, B11, B12, B21, B22)
        A11, A12, A21, A22 = normalize2by2(A11, A12, A21, A22)

    # solve linear equation for bwd[0]
    bwd[:, 0] = linearSolveBwd(A11, A12, A21, A22)

    # compute other points from bwdd0:
    bwd[:, N-1] = A[:, N-1]*bwd[:, 0] + input_img[:, N-1]

    for y in range(N-2, 0, -1):
        bwd[:, y] = A[:, y]*bwd[:, y+1] + input_img[:, y]
    return bwd


def compute_circularFwd(bwd, B, C):
    M, N = np.shape(bwd)
    fwd = bwd*0

    A11 = B[:, 0]*0 + 1
    A12 = 0*A11
    A21 = 0*A11
    A22 = 1*A11

    for y in range(1, N+1):
        B11 = B[:, y % N]
        B12 = -bwd[:, y % N]
        B21 = A11*0
        B22 = C[:, y % N]

        A11, A12, A21, A22 = multiply2by2(
            A11, A12, A21, A22, B11, B12, B21, B22)
        A11, A12, A21, A22 = normalize2by2(A11, A12, A21, A22)
    # solve linear equation for fwd[0]
    fwd[:, 0] = linearSolveBwd(A11, A12, A21, A22)

    # compute other poitns from fwd[0]
    for y in range(1, N):
        fwd[:, y] = (bwd[:, y] + C[:, y]*fwd[:, y-1])/B[:, y]

    return fwd


def tikhonov1D_y(input_img, W, mode='truncate'):
    if mode == 'truncate':
        M, N = np.shape(W)
        A = W*0
        B = W*0
        C = W*1

        # compute B
        B[:, N-1] = 1 + W[:, N-1]
        for y in range(N-2, -1, -1):
            B[:, y] = 1 + W[:, y] + W[:, y+1] - W[:, y+1]**2/(B[:, y+1])

        # compute A
        A[:, 0:-1] = W[:, 1:]/B[:, 1:]
        A[:, -1] = 0

        # bwd filter
        bwd = input_img*1
        for y in range(N-2, -1, -1):
            bwd[:, y] = A[:, y]*bwd[:, y+1] + input_img[:, y]

        fwd = bwd/B
        for y in range(1, N):
            fwd[:, y] = (bwd[:, y] + C[:, y]*fwd[:, y-1])/B[:, y]

        return fwd

    elif mode == 'circular':
        M, N = np.shape(W)
        A = W*0
        B = W*0
        C = W*1

        B = compute_circularB(W)
        A = np.roll(W, -1, axis=1)/np.roll(B, -1, axis=1)

        bwd = compute_circularBwd(input_img, A)
        fwd = compute_circularFwd(bwd, B, C)
        return fwd


def tikhonov1D_x(input_img, W, mode='truncate'):
    img = np.transpose(input_img)
    Wx = np.transpose(W)
    res = tikhonov1D_y(img, Wx, mode=mode)
    return np.transpose(res)


def admm_method_gastal(input_img, sigma_s, sigma_r, alpha_r, tol=1e-3, rho=10, bound=0, channels=1, mode='truncate'):

    luminance = vis.get_luminance(input_img)

    mono_image = luminance*0
    ax, ay, dt_x, dt_y, Wx, Wy, success = getDomainTransformCoefficients(
        luminance, sigma_s, sigma_r, alpha_r, tol=tol, mode=mode)
    max_it = 10000
    output_image = input_img*0
    output_data = dict()

    if success == False:
        output_data["success"] = False
        return output_image, output_data  # failed in acquiring coefficients

    for c in range(channels):
        if channels == 1:
            mono_image = luminance*1
        else:
            mono_image = input_img[:, :, c]*1

        X = mono_image*1
        Y = mono_image*1
        U = X*0
        it = 0
        err = 1.0
        while (err > tol) and (it < max_it):
            X = tikhonov1D_x((mono_image + rho*Y - U) /
                             (1+rho), 2*Wx/(1+rho), mode=mode)
            Y = tikhonov1D_y((mono_image + rho*X + U) /
                             (1+rho), 2*Wy/(1+rho), mode=mode)
            U = U + rho*(X-Y)
            it += 1
            # check error each 10 iterations

            if it % 10 == 0:
                err = np.linalg.norm(X - Y)/np.linalg.norm(mono_image)
        output_image[:, :, c] = X*1

    if channels == 1:
        for c in range(3):
            output_image[:, :, c] = X*1

    if it >= max_it:
        output_data["success"] = False
    else:
        output_data["success"] = True
    output_data["admm_it"] = it

    return output_image, output_data

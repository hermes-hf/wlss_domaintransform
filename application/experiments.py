import numpy as np
import source.vis as vis
import source.wlss as wlss
import source.domain_transform_filter as dtf
import sys
import os
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import pandas as pd
import itertools
import time

def check_outputfile():
    #check if file exists
    if not os.path.isfile('outputs.csv'):
        csv_header = pd.DataFrame([], columns = ['image','sigma_s','sigma_r','alpha_r', 'psnr', 'ssim', 
        'gastal_admm_it', 'farbman_admm_it', 'success','tol'])
        csv_header.to_csv("outputs.csv", index = False)

def generate_inputs(img_paths):
    input_list = []
    for path in img_paths:
        input_img = vis.load_img_rgb('data/'+path)
        N,M = np.shape(input_img[:,:,0])
        diag_len = np.sqrt(N**2 + M**2)
        
        sigma_s_list = list(np.linspace(1, diag_len/3, 10))
        sigma_r_list = list(np.linspace(1,255,15))
        alpha_r_list = list(np.linspace(0.5,1,11))
        input_list = input_list + list(itertools.product([path], sigma_s_list, sigma_r_list, alpha_r_list))

    return pd.DataFrame(input_list, columns = ['image','sigma_s','sigma_r','alpha_r'])

def apply_methods(row, input_img, tol, channels, rho):
    output_img_gastal, outs_gastal = dtf.admm_method_gastal(input_img, row['sigma_s'], row['sigma_r'], row['alpha_r'], 
                                                                            tol=tol, rho=rho, channels=channels)
    output_img_farbman, outs_farbman = wlss.admm_method_farbman(input_img,  row['sigma_s'], row['sigma_r'], row['alpha_r'], 
                                                                tol=tol, rho=rho, channels=channels)
    img_psnr = PSNR(output_img_farbman[:,:,0], output_img_gastal[:,:,0], data_range=255)
    img_ssim = SSIM(output_img_farbman[:,:,0], output_img_gastal[:,:,0], data_range=255)

    results = {'image' : row['image'],'sigma_s' : row['sigma_s'], 'sigma_r' : row['sigma_r'], 'alpha_r' : row['alpha_r'],
            'psnr' : img_psnr, 'ssim' : img_ssim, 'gastal_admm_it' : outs_gastal['admm_it'],
    'farbman_admm_it' : outs_farbman['admm_it'], 'success' : outs_farbman['success']*outs_gastal['success'], 'tol' : tol}
    return results

def save_rows(result_list):
    df = pd.DataFrame(result_list, columns = ['image','sigma_s','sigma_r','alpha_r', 'psnr', 'ssim', 
        'gastal_admm_it', 'farbman_admm_it', 'success', 'tol'] )
    df.to_csv('outputs.csv', mode='a', header=False, index=False)
    return

def get_remaining_inputs(img_paths):
    df_inputs = generate_inputs(img_paths)
    df_existing_inputs = pd.read_csv("outputs.csv")[['image','sigma_s','sigma_r','alpha_r']]
    return df_inputs[~df_inputs.isin(df_existing_inputs)].dropna()

def main():
    img_paths = os.listdir("data/")
    tol = 1e-3
    rho = 20
    channels = 1

    check_outputfile()
    df_inputs = get_remaining_inputs(img_paths)
    dt_average = 0
    
    for img_name, group in df_inputs.groupby('image'):
        df_image = df_inputs[df_inputs['image']==img_name]
        input_img = vis.load_img_rgb('data/'+img_name)

        t = 1
        result_list = []
        for index, row in df_image.iterrows():
            start_time = time.time()
            result = apply_methods(row, input_img, tol, channels, rho)
            dt = (time.time() - start_time) #dt in seconds
            if dt_average==0:
                dt_average = dt
            else:
                dt_average = 0.5*dt_average + 0.5*dt #SES filter in dt to estimate 'average' dt
            result_list.append(result.values())

            if t%10==0:
                dt_days = dt_average/(60*60*24)
                if (t%100==0) or (t == len(df_image)) :
                    save_rows(result_list)
                    result_list = []
                    print("Parsed image {}, {}/{}, estimated remaining time = {} days".format(row['image'], t, len(df_image), round((len(df_image)-t)*dt_days, 2)))

                else:
                    print("Parsed image {}, {}/{}, estimated remaining time = {} days".format(row['image'], t, len(df_image), round((len(df_image)-t)*dt_days, 2)))
                    
            t+=1
            



if __name__ == "__main__":
    main()
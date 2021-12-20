import os,shutil

def Extract_Files_From_Prep(sourse_path, target_folder, keyword_for_replace='-T1w.anat'):
    """[This Python script is used to copy the various preprocessed resulting images into the specified folder.]

    Args:
        sourse_path ([str]): [The path of folder which contains preprocessed images.]
        target_folder ([str]): [The path of target folder where you can reorganize the resulting images.]
        keyword_for_replace (str, optional): [Filename suffix]. Defaults to '-T1w.anat'.
    """
    
    for home, dirs, files, in os.walk(sourse_path):
        for file_name in files:
            if '_nonlin_brain.nii.gz' in file_name:
                name = home.split('/')[-1].replace(keyword_for_replace,'-nonlin_brain.nii.gz')
                sourse_file_path = os.path.join(home, file_name)
                nonlinear_brain_folder_path = os.path.join(target_folder, 'nonlin_brain/')
                if not os.path.exists(nonlinear_brain_folder_path):
                    os.makedirs(nonlinear_brain_folder_path)
                target_file_path = os.path.join(nonlinear_brain_folder_path, name)
                print(home, file_name, name)
                print(sourse_file_path, target_file_path)
                shutil.copyfile(sourse_file_path, target_file_path)


            elif '_lin.nii.gz' in file_name:
                name = home.split('/')[-1].replace(keyword_for_replace,'-lin.nii.gz')
                sourse_file_path = os.path.join(home, file_name)
                linear_folder_path = os.path.join(target_folder, 'linear/')
                if not os.path.exists(linear_folder_path):
                    os.makedirs(linear_folder_path)
                target_file_path = os.path.join(linear_folder_path, name)
                print(home.split('/')[-1],file_name)
                print(sourse_file_path, target_file_path)
                shutil.copyfile(sourse_file_path, target_file_path)


            elif '_nonlin.nii.gz' in file_name:
                name = home.split('/')[-1].replace(keyword_for_replace,'-nonlin.nii.gz')
                sourse_file_path = os.path.join(home, file_name)
                nonlinear_folder_path = os.path.join(target_folder, 'nonlinear/')
                if not os.path.exists(nonlinear_folder_path):
                    os.makedirs(nonlinear_folder_path)
                target_file_path = os.path.join(nonlinear_folder_path, name)
                print(home.split('/')[-1], file_name)
                print(sourse_file_path, target_file_path)
                shutil.copyfile(sourse_file_path, target_file_path)


if __name__=='__main__':
    sourse_path = "/data/brain_age_estimation/"
    target_folder = "/data/brain_age_estimation_prep-org/"
    Extract_Files_From_Prep(sourse_path, target_folder,'_T1w.anat')
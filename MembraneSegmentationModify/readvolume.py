import os
import numpy as np
import cv2
import h5py
from PIL import Image

def readvolume(path, n=1, fun=lambda x: x):
    """
    Read a volume
    
    Parameters:
        path (str): file or directory path.
        n (int): replication factor along the z-axis. Default is 1.
        fun (callable): a function to apply to each slice of the volume. Default is identity.
        
    Returns:
        data (ndarray): the loaded volume data.
        ChunkSize (tuple or None): chunk size if H5 dataset provides it, else None.
    """
    ChunkSize = None

    # Check if path is a file or folder
    if os.path.isfile(path):
        # It's a file
        ext = os.path.splitext(path)[1].lower()
        if ext == '.tif':
            # Multipage TIFF
            # We can use PIL or tifffile
            # Here, using PIL for simplicity:
            img = Image.open(path)
            # Count how many frames
            frames = []
            try:
                while True:
                    frame = np.array(img)
                    frames.append(frame)
                    img.seek(img.tell()+1)
            except EOFError:
                pass
            
            num_images = len(frames)
            # Determine data type and shape
            first_frame = frames[0]
            h, w = first_frame.shape[:2]
            
            if first_frame.ndim == 2:
                # Grayscale
                # Replicate each slice 'n' times
                data = np.zeros((h, w, num_images*n), dtype=first_frame.dtype)
                for k in range(num_images):
                    slice_data = frames[k]
                    # replicate along z
                    data[:,:,k*n:(k+1)*n] = slice_data[:,:,None].repeat(n, axis=2)
            else:
                # Color image (e.g. RGB)
                # Assume shape (H,W,3)
                data = np.zeros((h, w, first_frame.shape[2], num_images*n), dtype=first_frame.dtype)
                for k in range(num_images):
                    slice_data = frames[k]
                    data[:,:,:,k*n:(k+1)*n] = slice_data[:,:,:,None].repeat(n, axis=3)
            
        elif ext == '.h5':
            # H5 file
            with h5py.File(path, 'r') as f:
                # Check for 'stack' dataset
                if 'stack' in f:
                    dset = f['stack']
                    ChunkSize = dset.chunks
                    # dset shape: assume (Z,Y,X) or something similar
                    # The original code does permute([2 3 1]) on '/stack', 
                    # meaning original likely (Z, ...). We need to adapt.
                    data = dset[...]
                    # If original MATLAB did permute([2 3 1]) this might mean:
                    # Original: (Z,Y,X)
                    # After permute: (Y,X,Z)
                    # We'll guess original is (Z,Y,X) -> after permute => (Y,X,Z)
                    data = np.transpose(data, (1,2,0))
                else:
                    # try '/volume/predictions'
                    dset = f['/volume/predictions']
                    # original code permutes data(1,:,:,:) -> (2 3 4 1)
                    # Suppose predictions shape is (C,Z,Y,X), we extract C=1 slice:
                    data_full = dset[...]
                    # extract first channel: data_full[0,...] 
                    data = data_full[0,...] # now shape (Z,Y,X)
                    data = np.transpose(data, (1,2,0)) # (Y,X,Z)
                    ChunkSize = dset.chunks
            num_images = data.shape[2]
            # If 'n' > 1, replicate each slice in z
            if n > 1:
                data = np.repeat(data, n, axis=2)
        else:
            raise ValueError("Unsupported file extension: {}".format(ext))
    
    else:
        # path is not a file, maybe a directory
        if os.path.isdir(path):
            filelist = [f for f in os.listdir(path) if f.lower().endswith(('.png','.tif'))]
            filelist.sort() # ensure sorted order
        else:
            # path might be a pattern
            dirpath = os.path.dirname(path)
            pattern = os.path.basename(path)
            filelist = [f for f in os.listdir(dirpath) if f.startswith(pattern)]
            filelist.sort()
            path = dirpath
        
        if len(filelist) == 0:
            raise FileNotFoundError("No images found in the provided path.")
        
        # Read the first image to get shape and dtype
        first_img = cv2.imread(os.path.join(path, filelist[0]), cv2.IMREAD_UNCHANGED)
        h, w = first_img.shape[:2]
        dtype = first_img.dtype
        
        # Check if color or grayscale
        if first_img.ndim == 3:
            # color
            data = np.zeros((h, w, first_img.shape[2], len(filelist)*n), dtype=dtype)
            for k, fname in enumerate(filelist):
                im_ = cv2.imread(os.path.join(path, fname), cv2.IMREAD_UNCHANGED)
                data[:,:,:,k*n:(k+1)*n] = im_[:,:,:,None].repeat(n, axis=3)
        else:
            # grayscale
            data = np.zeros((h, w, len(filelist)*n), dtype=dtype)
            for k, fname in enumerate(filelist):
                im_ = cv2.imread(os.path.join(path, fname), cv2.IMREAD_UNCHANGED)
                data[:,:,k*n:(k+1)*n] = im_[:,:,None].repeat(n, axis=2)
        
        num_images = len(filelist)*n
    
    # Apply fun to each slice
    # If data is (H,W,Z) or (H,W,C,Z), apply fun to each slice along Z
    if data.ndim == 3:
        # (H,W,Z)
        for k in range(data.shape[2]):
            data[:,:,k] = fun(data[:,:,k])
    elif data.ndim == 4:
        # (H,W,C,Z)
        for k in range(data.shape[3]):
            # apply fun to each slice maybe on a single channel or all channels
            # The original code only did data(:,:,k)=fun(data(:,:,k))
            # If we have multiple channels, apply fun channel-wise:
            # If fun expects single-channel, might need to handle differently.
            # We'll assume fun can handle the full array (H,W,C):
            data[:,:,:,k] = fun(data[:,:,:,k])
    
    return data, ChunkSize

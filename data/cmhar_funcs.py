import numpy as np



def cmharCalcWindow(data:np.ndarray) -> list:
    data_len        = len(list(data[0]))
    data_remains    = True
    ret_window      = []
    last_idx        = 0

    while data_remains:
        if data_len - last_idx >= 150:
            ret_window.append((last_idx, last_idx + 149))
            last_idx += 75
        
        elif data_len - last_idx > 0:
            less_offset = 150 - (data_len - last_idx)
            if last_idx - less_offset < 0:
                ret_window.append((0, last_idx - less_offset + 149))
            else:
                ret_window.append((last_idx - less_offset, last_idx - less_offset + 149))
            data_remains = False

        else:
            data_remains = False

    return ret_window



def cmharCalcRms(data:np.ndarray) -> list:
    ret_val = []

    # Calculate window
    window = cmharCalcWindow(data)

    # Calculate RMS for every window
    for win in window:
        val_qw = 0.
        val_qx = 0.
        val_qy = 0.
        val_qz = 0.

        for i in range(win[0], win[1] + 1):
            val_qw += (data[1][i]**2.0)/150.0
            val_qx += (data[2][i]**2.0)/150.0
            val_qy += (data[3][i]**2.0)/150.0
            val_qz += (data[4][i]**2.0)/150.0

        val_qw = np.sqrt(val_qw)
        val_qx = np.sqrt(val_qx)
        val_qy = np.sqrt(val_qy)
        val_qz = np.sqrt(val_qz)

        ret_val.append([val_qw, val_qx, val_qy, val_qz])

    return ret_val



def cmharCalcCovariance(data:np.ndarray) -> list:
    ret_val = []

    # Calculate window
    window = cmharCalcWindow(data)

    # Calculate covariance matrix
    for win in window:
        val_wx = np.cov(data[1][win[0] : win[1] + 1], data[2][win[0] : win[1] + 1])
        val_wy = np.cov(data[1][win[0] : win[1] + 1], data[3][win[0] : win[1] + 1])
        val_wz = np.cov(data[1][win[0] : win[1] + 1], data[4][win[0] : win[1] + 1])
        val_xy = np.cov(data[2][win[0] : win[1] + 1], data[3][win[0] : win[1] + 1])
        val_xz = np.cov(data[2][win[0] : win[1] + 1], data[4][win[0] : win[1] + 1])
        val_yz = np.cov(data[3][win[0] : win[1] + 1], data[4][win[0] : win[1] + 1])

        cov_mat = np.array([
            [val_wx[0][0], val_wx[0][1], val_wy[0][1], val_wz[0][1]],
            [val_wx[0][1], val_xy[0][0], val_xy[0][1], val_xz[0][1]],
            [val_wy[0][1], val_xy[0][1], val_yz[0][0], val_yz[0][1]],
            [val_wz[0][1], val_xz[0][1], val_yz[0][1], val_yz[1][1]]
        ])

        eigval = np.linalg.eigvals(cov_mat)

        ret_val.append([
            val_wx[0][0], val_wx[0][1], val_wy[0][1], val_wz[0][1], 
            val_xy[0][0], val_xy[0][1], val_xz[0][1],
            val_yz[0][0], val_yz[0][1],
            val_yz[1][1]
        ] + list(eigval))

    return ret_val



def cmharCalcMedian(data:np.ndarray) -> list:
    ret_val = []

    # Calculate window
    window = cmharCalcWindow(data)

    # Calculate variance for every window
    for win in window:
        val_qw = np.median(data[1][win[0] : win[1] + 1])
        val_qx = np.median(data[2][win[0] : win[1] + 1])
        val_qy = np.median(data[3][win[0] : win[1] + 1])
        val_qz = np.median(data[4][win[0] : win[1] + 1])

        ret_val.append([val_qw, val_qx, val_qy, val_qz])

    return ret_val
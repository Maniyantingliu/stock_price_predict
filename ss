 n_x = scaler.fit_transform(np.array(new_data)[i:i+time_step, -1])
            n_y = scaler.fit_transform(np.array(new_data)[i+time_step:i+time_step+pred_step, -1])
            X.append(np.concatenate((np.array(new_data)[i:i+time_step, :-1], n_x[:,np.newaxis]), axis=1))
            Y.append(n_y)
import matplotlib.pyplot as plt
import pandas as pd

def plot_pdf_3D(self, df:pd.DataFrame, label_column:str):
    # train the model
    df_without_label = df.drop(columns=[label_column])
    df_cov = df_without_label.values

    self.fit(df_without_label, list(df[label_column]))

    feats = df_without_label.columns
    u_labels = df[label_column].unique()

    for i1 in range(len(feats)-1):
        for i2 in range(i1+1, len(feats)-1):
            feat_1 = feats[i1]
            feat_2 = feats[i2]
            for i_label, label in enumerate(u_labels):
                min_value_1 = min(df[feat_1])
                max_value_1 = max(df[feat_1])
                min_value_2 = min(df[feat_2])
                max_value_2 = max(df[feat_2])
                N = 0.1
                X = np.arange(min_value_1, max_value_1, N)
                Y = np.arange(min_value_2, max_value_2, N)
                X, Y = np.meshgrid(X, Y)
                Z = np.empty(X.shape)
                mu = self.label_to_mean_v[label]
                Sigma = compute_covariance_matrix(df_cov)
                for i in range(Z.shape[0]):
                    for j in range(Z.shape[1]):
                        x = np.array([[X[i][j]],[Y[i][j]]])
                        Z[i][j] = self._likelihood_multivariate(x, mu, Sigma)
                
                ax = plt.axes(projection='3d')
                ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                                cmap='viridis', edgecolor='none')
                ax.set_title(label)
                ax.set_xlabel(feat_1)
                ax.set_ylabel(feat_2)
                ax.set_zlabel('Discriminant')
                plt.show()
                
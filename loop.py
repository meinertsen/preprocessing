from inspect import getmembers, isclass

from sklearn.model_selection import train_test_split
X_train, X_test, L_train, L_test = train_test_split(X, L, test_size = 0.5, random_state = 7)

def prep(X_train, X_test, L_train, L_test):
    from scipy import sparse
    from sklearn import preprocessing
    for ppname,preprocess in [(i,j) for i,j in getmembers(preprocessing,isclass)]:
        if preprocess.__module__  == 'sklearn.preprocessing.data':
            prep = preprocess().fit(X_train)
            X_train_prep = prep.transform(X_train)
            X_test_prep = prep.transform(X_test)
                if not sparse.issparse(X_train_prep):
                    # do something
                    # Dim redux (PCA, manifold)
                    
# Version includes PCA (default 2 components) and IsoMap (with variable k) as dimensionality reducers and KNN (with variable k)
Scalers = ['MaxAbsScaler','MinMaxScaler','StandardScaler','Normalizer','RobustScaler']
def prep(X_train, X_test, L_train, L_test):
    results=[]
    from scipy import sparse
    for ppname,preprocess in [(i,j) for i,j in getmembers(preprocessing,isclass)]:
#        if preprocess.__module__  == 'sklearn.preprocessing.data':
        if ppname in Scalers:
            prep = preprocess().fit(X_train)
            X_train_prep = prep.transform(X_train)
            X_test_prep = prep.transform(X_test)
            if not sparse.issparse(X_train_prep):
                for n in range(5,11):
                    from sklearn import manifold
                    iso = manifold.Isomap(n_neighbors = n, n_components = 2)
                    iso = iso.fit(X_train_prep)
                    data_train = iso.transform(X_train_prep)
                    data_test = iso.transform(X_test_prep)
                    for k in range(1,16):
                        from sklearn.neighbors import KNeighborsClassifier
                        knn = KNeighborsClassifier(n_neighbors = k)
                        knn.fit(data_train, L_train)
                        results.append([ppname, "ISO "+str(n), k, knn.score(data_test, L_test)])
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2, random_state=1)
                pca.fit(X_train_prep)
                data_train = pca.transform(X_train_prep) 
                data_test = pca.transform(X_test_prep)  
                for k in range(1,16):
                    from sklearn.neighbors import KNeighborsClassifier
                    knn = KNeighborsClassifier(n_neighbors = k)
                    knn.fit(data_train, L_train)
                    results.append([ppname, "PCA",k, knn.score(data_test, L_test)])
    return results                

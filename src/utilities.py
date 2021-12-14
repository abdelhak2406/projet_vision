import pickle5 as pickle

def openPkl(filename,pathopen):
    with  open(pathopen+filename,"rb") as file:
        return pickle.load(file)

def savePkl(objname,filename,pathsave):
    """
    objname : nom de l'objet qu'on veut sauvegarder
    filename : sous quel nom on veut le sauvegarder (ajouter .pkl au nom )
    pathsave : path ou on vas mettre filename
    """
    with  open(pathsave+filename,"wb") as file:
        pickle.dump(objname,file,pickle.HIGHEST_PROTOCOL)


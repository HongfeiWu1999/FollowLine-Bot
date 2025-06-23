import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import NearestCentroid
from sklearn.cluster import DBSCAN
import math

import pickle

"""
    Se utiliza cv2 para la lectura y procesamiento de la imagen, 
    por lo tanto las imagenes leidas estaran en formato BGR.

    Etiqueta de la imagen:
     -> marca = 0
     -> suelo = 1
     -> linea = 2
"""
"""
imNp = cv2.imread('images/muestras/ball_distance_15cm.jpg') => 15cm, 40cm,70cm
imNp = cv2.medianBlur(imNp, 9) => Eliminamos el ruido
normalizedImg = imm.normalizeImage(imNp)

red = [0,0,255]
green = [0,255,0]

labels = imm.predictBall(normalizedImg)

ballContour = imm.findContours(labels,0) => Buscamos el contorno de la bola

ellip, lkp, desc = imm.getDescription(labels, ballContour) => Obtenemos el elipse con sus correspondientes informacion

cen, ejes, angulo = np.array(ellip[0]),np.array(ellip[1]),ellip[2]

u1_u2 = np.mean(ejes)*1.3 => El diametro de la bola en la imagen expresado en pixeles.

d = 7.5 => El diametro de la bola real 7.5cm

Z = 15 => La distancia de la camara con la bola. 15cm, 40cm, 70cm

f = Z * u1_u2 / d => Funcion f a aproximar
"""

class ImageManager:
  
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.scale = 0.8
        self.color = 0
        self.thick = 2
        self.orb = cv2.ORB_create()
        self.image_clf = self.getImageClf()
        self.mark_clf = self.getMarkClf()
        self.ball_clf = self.getBallClf()
        self.dbscan = DBSCAN(eps=20, min_samples=5)
        self.f = 2000
        # Contador para analisis de datos
        self.count = 0
        # Historial de entrada y salida
        self.start_point = None
        self.end_point = None
        # Centroid del contorno de la linea
        self.line_centroid = None
        # Punto medio de cada salida posible
        self.mean_end_points = None
        # Angulo indicado por la flecha encontrado
        self.arrowIndicated = False
        
    def getImageClf(self):
        # Leo los datos de entrenamiento
        with open("samples/image_Xtrain.pkl", "rb") as file:
            X_train = pickle.load(file)
            
        with open("samples/image_yTrain.pkl", "rb") as file:
            y_train = pickle.load(file)
            
        # Entrena el clasificador con los datos de entrenamiento
        clf_image = LinearDiscriminantAnalysis()
        clf_image.fit(X_train,y_train)
        return clf_image
    
    def getMarkClf(self):
        # Leo los datos de entrenamiento
        with open("samples/mark_Xtrain.pkl", "rb") as file:
            X_train = pickle.load(file)
            
        with open("samples/mark_yTrain.pkl", "rb") as file:
            y_train = pickle.load(file)
            
        clf_mark = SVC(kernel='linear')
        clf_mark.fit(X_train,y_train)
        return clf_mark
    
    def getBallClf(self):
        # Leo los datos de entrenamiento
        with open("samples/ball_Xtrain.pkl", "rb") as file:
            X_train = pickle.load(file)
            
        with open("samples/ball_yTrain.pkl", "rb") as file:
            y_train = pickle.load(file)
        
        clf_ball= LinearDiscriminantAnalysis()
        clf_ball.fit(X_train,y_train)
        return clf_ball
        
    def getImageMask(self, image, color):
        return np.all(image == color, axis=-1)

    def getImage(self, img, color):
        mask = self.getImageMask(img, color)
        image = np.zeros_like(img)
        image[mask] = 255
        return image

    def normalizeImage(self, image):
        sum = np.sum(image, axis=2)
        return np.nan_to_num(image / sum[:, :, np.newaxis])

    def predict(self, image):
        X = image.reshape((-1,3))
        y_pred = self.image_clf.predict(X)
        return y_pred.reshape(image.shape[:2])
    
    def predictBall(self, image):
        X = image.reshape((-1,3))
        y_pred = self.ball_clf.predict(X)
        return y_pred.reshape(image.shape[:2])
    
    def findExitPoints(self, width, height, lineContour):
        top_points = lineContour[lineContour[:, 0, 1] == 0]
        left_points = lineContour[lineContour[:, 0, 0] == 0]
        right_points = lineContour[lineContour[:, 0, 0] == width - 1]
        bottom_points = lineContour[lineContour[:, 0, 1] == height - 1]
        
        return top_points, left_points, right_points, bottom_points
    
    def findLineContour(self, labels):
        contImg = (labels==2).astype(np.uint8)*255
        contList,hier = cv2.findContours(contImg,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        contList = [cont for cont in contList if cv2.contourArea(cont) > 1000]
        length = len(contList)
        if length == 0:
            return None
        elif length == 1:
            return contList[0]
        else:
            # Calcular distancia entre el punto de salida y cada contorno
            distances = [abs(cv2.pointPolygonTest(cont, self.end_point, True)) for cont in contList]
            # Seleccionar contorno con la menor distancia al punto de salida
            return contList[np.argmin(distances)]
    
    def findContours(self, labels, label):
        contImg = (labels==label).astype(np.uint8)*255
        contList,hier = cv2.findContours(contImg,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        return max(contList, key=cv2.contourArea) if len(contList) != 0 else None
    
    def findNotPartContours(self, labels, label):
        contImg = (labels!=label).astype(np.uint8)*255
        contList,hier = cv2.findContours(contImg,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        return [contour for contour in contList if cv2.contourArea(contour) > 1500]
        
    def getLineDirection(self, lineContour, notLineContList):
        lineCount = len(notLineContList) - 1
        if lineCount == 1:
            convHull = cv2.convexHull(lineContour, returnPoints=False)
            convDefs = cv2.convexityDefects(lineContour, convHull)

            if convDefs is not None:
                cnvDef = convDefs[np.argmax(convDefs[:, 0, 3])].tolist()
                [start,end,mid,length] = cnvDef[0]
                
                if length <= 1000: 
                    return "Linea recta"
                else:
                    start_point = lineContour[start][0]
                    end_point = lineContour[end][0]
                    
                    if start_point[1] < end_point[1]:
                        init_x = start_point[0]
                        end_x = end_point[0]
                    else:
                        init_x = end_point[0]
                        end_x = start_point[0]
                        
                    if init_x > end_x:
                        return "Curva hacia derecha"
                    else:
                        return "Curva hacia izquierda"
            else:
                return "Linea recta"
        elif lineCount == 2:
            return "Cruce con dos salidas"
        else:
            return "Cruce con tres salidas"
        
    def getLineDetails(self, image, lineContour, notLineContList):
        # Marcar el tipo de salida
        lineDirection = self.getLineDirection(lineContour, notLineContList)
        cv2.putText(image, lineDirection, (10, 30), self.font, self.scale, self.color, self.thick)
        
        # Los puntos que cruzan con los border de la imagen de la linea
        top_points, left_points, right_points, bottom_points = self.findExitPoints(image.shape[1], image.shape[0], lineContour)
        
        # Si no tenemos ningun punto de entrada, la entra seria los puntos que cruza con el borde inferior
        if self.start_point is None or self.end_point is None:
            self.start_point = np.mean(bottom_points[:,0,:].T, axis=1).astype(int)
            self.end_point = np.mean(np.concatenate((top_points, left_points, right_points))[:,0,:].T, axis=1).astype(int)
        else:
            # Agrupamos los puntos
            points = np.concatenate((top_points, left_points, right_points, bottom_points))[:,0,:]
            groups = self.dbscan.fit_predict(points)
            
            # Sacamos la media de cada grupo de puntos
            mean_points = np.array([np.mean(points[groups == group].T, axis=1) for group in np.unique(groups)]).astype(int)
            # Encontramos el indice del punto mas proximo a la entrada anterior
            start_point_idx = np.argmin([self.getMinDistance(self.start_point, point) for point in mean_points])
            # Establecemos los puntos de entrada de la linea
            self.start_point = mean_points[start_point_idx]
            
            # Establecemos los puntos medios de las salidas para el uso posterior del angulo con la flecha
            self.mean_end_points = np.delete(mean_points, start_point_idx, axis=0)
            if self.mean_end_points.shape[0] != 0:
                # Encontramos el indice del punto mas proximo a la salida anterior
                end_point_idx = np.argmin([self.getMinDistance(self.end_point, point) for point in self.mean_end_points])
                # Establecemos los puntos de salida de la linea
                self.end_point = self.mean_end_points[end_point_idx]
            
        # Convertimos el array de puntos en tuplas
        self.start_point = tuple(self.start_point)
        self.end_point = tuple(self.end_point)
        
        # Marcamos el punto de entrada y salida
        # cv2.circle(image, self.start_point, 5, (0,0,0), 5)
        cv2.circle(image, self.end_point, 5, (0,0,255), 5)
        
        # Calcular momentos del contorno
        moments = cv2.moments(lineContour)

        # Calcular coordenadas x e y del centroide de la linea
        x = int(moments['m10'] / moments['m00'])
        y = int(moments['m01'] / moments['m00'])
        
        # Establecemos el centroide y los puntos medios de las salidas
        self.line_centroid = (x,y)
        
        # Calculamos el error de la desviacion
        error = self.findLineDeviation(image.shape[1],image.shape[0])
        cv2.putText(image, str(error), (10, 60), self.font, self.scale, self.color, self.thick)
        
        return error
       
    def drawContourMark(self, image, labels, markContour, notLineContList):
        # Comprobamos que existe el contorno de la marca y que la marca sea de tamano suficiente
        if markContour is not None and len(markContour) > 5 and cv2.contourArea(markContour) > 500:
            # Obtenemos el elipse y las descripciones de la marca
            ellip, lkp, desc = self.getDescription(labels, markContour)
            
            if desc is not None:
                lineCount = len(notLineContList) - 1
                if  lineCount == 1:
                    typeMark = self.mark_clf.predict(desc)

                    if typeMark == 0:
                        mark = "Man"
                    elif typeMark == 1:
                        mark = "Stair"
                    elif typeMark == 2:
                        mark = "Telephone"
                    else:
                        mark = "Woman"
                    cv2.putText(image, mark, (10, 90), self.font, self.scale, self.color, self.thick)
                else:
                    cv2.putText(image, "Arrow", (10, 90), self.font, self.scale, self.color, self.thick)
                    
                    # Obtenemos el angulo de la flecha
                    arrow_angle = self.getArrowAngle(image, markContour)
                    if arrow_angle is not None:
                        # Calculamos sus angulos correspondientes de cada salida, por ello, utilizamos el centro del contorno de la linea
                        end_point_angles = [self.getPointAngle(self.line_centroid, end_point) for end_point in self.mean_end_points]
                        
                        # Sacamos la salida con el angulo mas proximo a la flecha
                        closest_exit_idx = np.argmin([min(abs(angle - arrow_angle), 360 - abs(angle - arrow_angle)) for angle in end_point_angles ])
                        # Establecemos los puntos de salida de la linea elegida acuerdo con el angulo de la flecha
                        self.end_point = self.mean_end_points[closest_exit_idx]
                    
            # Pintamos el resultado de la deteccion
            cv2.drawContours(image, markContour, -1, (0,255,0),2)
            cv2.ellipse(image, ellip, (255,0,255),2)
            cv2.drawKeypoints(image.astype(np.uint8), lkp, None, color=(0,255,0),flags = cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)     
            
    def getDescription(self, labels, contour):
        # Ajusto elipse para identificar la orientacion
        ellip = cv2.fitEllipse(contour)
        cen, ejes, angulo = np.array(ellip[0]),np.array(ellip[1]),ellip[2]
        kp = cv2.KeyPoint(cen[0],cen[1], np.mean(ejes)*1.3, angulo-90)
        
        # Describe la region
        lkp, desc = self.orb.compute(labels.astype(np.uint8), [kp])
        return ellip, lkp, desc
    
    def getPointAngle(self, start_point, end_point):
        # Calculamos el ángulo entre los puntos A y B
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        angle_degrees = cv2.fastAtan2(dy, dx)

        # Ajustamos el ángulo a rango de 0 a 360 grados
        if angle_degrees < 0:
            angle_degrees += 360
            
        # Convertimos al angulo de sentido antihorario estandar    
        angle_degrees = 360 - angle_degrees
        return angle_degrees     
    
    def findArrowTip(self, points, convex_hull):
        # Encontrar la punta y la cola de la flecha
        length = len(points)
        indices = np.setdiff1d(range(length), convex_hull)
        for i in range(2):
            j = indices[i] + 2
            if j > length - 1:
                j = length -j
            p = j + 2
            if p > length - 1:
                p = length -p
            if np.all(points[j] == points[indices[i-1]-2]):
                return points[j] if not self.isPointsNeighbor(points[j], points[p]) else None
            
    def getArrowAngle(self, image, arrowContour):
        peri = cv2.arcLength(arrowContour, True)
        approx = cv2.approxPolyDP(arrowContour, 0.025 * peri, True)
        hull = cv2.convexHull(approx, returnPoints=False)
        sides = len(hull)
        if 6 > sides > 3 and sides + 2 == len(approx):
            # Buscamos el vertice que es la punta de la flecha
            arrow_tip = self.findArrowTip(approx[:,0,:], hull.squeeze())
            if arrow_tip is not None:
                # Calculaamos el momento del contorno
                moments = cv2.moments(arrowContour)

                # Calcular coordenadas x e y del centroide
                x = int(moments['m10'] / moments['m00'])
                y = int(moments['m01'] / moments['m00'])
                return self.getPointAngle((x,y), arrow_tip)
            
        return None
    
    def getMinDistance(self, point1, point2):
        return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
    
    def isPointsNeighbor(self, point1, point2, threshold=20):
        # Mira si la distancia entre dos puntos cumple el umbral establecido
        return self.getMinDistance(point1, point2) < threshold
    
    def findLineDeviation(self, width, height):
        # Calculamos los vectores entre tres puntos
        vector1 = (0, height - height // 2)
        vector2 = (self.end_point[0] - width // 2, self.end_point[1] - height // 2)

        # Calculamos el producto escalar de los dos vectores
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

        # Calculamos las magnitudes de los dos vectores.
        magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
        magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)

        # Calculamos el angulo entre los dos vectores
        angle = math.acos(dot_product / (magnitude1 * magnitude2))

        # Convertimos de radianes a grados
        angle_degrees = math.degrees(angle)
        
        # Normalizamos el resultado a 0-1
        error = min(1 , ((180 - angle_degrees) / 180) * 1.6)
        
        # Decidimos girar hacia izquierda o hacia derecha
        if self.end_point[0] < width // 2:
            return -error
        
        return error
    
    def recoverLineAfterAvoidObstacle(self, image, lineContour):
        # Los puntos que cruzan con los border de la imagen de la linea
        top_points, left_points, right_points, bottom_points = self.findExitPoints(image.shape[1], image.shape[0], lineContour)
        
        # Agrupamos los puntos
        points = np.concatenate((top_points, left_points, right_points, bottom_points))[:,0,:]
        groups = self.dbscan.fit_predict(points)
        
        # Sacamos la media de cada grupo de puntos
        mean_points = np.array([np.mean(points[groups == group].T, axis=1) for group in np.unique(groups)]).astype(int)
        
        # Encontramos el indice del punto mas proximo al borde izquierdo que sera la entrada
        start_point_idx = np.argmin(mean_points[:,0])
        # Establecemos los puntos de entrada de la linea
        self.start_point = mean_points[start_point_idx]

        # Establecemos los puntos medios de las salidas para el uso posterior del angulo con la flecha
        self.mean_end_points = np.delete(mean_points, start_point_idx, axis=0)
        if self.mean_end_points.shape[0] != 0:
            # Encontramos el indice del punto mas proximo al borde derecho
            end_point_idx = np.argmax(self.mean_end_points[:,0])
            # Establecemos los puntos de salida de la linea
            self.end_point = self.mean_end_points[end_point_idx]
            
        # Convertimos el array de puntos en tuplas
        self.start_point = tuple(self.start_point)
        self.end_point = tuple(self.end_point)
        
    def printDistance(self, image, labels):
        # Obtenemos el contorno de la bola
        ballContour = self.findContours(labels,0)
        if ballContour is not None and len(ballContour) > 5 and cv2.contourArea(ballContour) > 300:
            # Sacamos las informacion del contrno
            ellip, lkp, desc = self.getDescription(labels, ballContour)
            cen, ejes, angulo = np.array(ellip[0]),np.array(ellip[1]),ellip[2]

            # Dibujamos el elipse del contorno
            cv2.ellipse(image, ellip, (0,255,0),2)

            # Aproximamos la distancia de la camara con la bola
            u1_u2 = np.mean(ejes) * 1.3
            d = 7.5

            # La distancia del objeto con la camara, truncandolo a dos decimales
            Z = round(self.f * d / u1_u2, 2)
            distance = "Esta a " + str(Z) + " cm de la camara"
            
            cv2.putText(image, distance, (10, 30), self.font, self.scale, self.color, self.thick)
            

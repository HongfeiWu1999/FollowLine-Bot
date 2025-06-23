from pyrobot.brain import Brain

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from image_manager import ImageManager

class BrainFollowLine(Brain):

    def setup(self):
        #---------------------------Simulated World-------------------------
        # self.image_sub = rospy.Subscriber("/image", Image, self.callback)
        # self.bridge = CvBridge()
        
        #---------------------------Real World------------------------------
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        
        #---------------------------Video Record---------------------------
        # Crear un objeto de escritura de video
        # self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # self.out = cv2.VideoWriter('videos/video_linea_marca.avi', self.fourcc, 30, (640, 260))
        
        #--------------------------Constants--------------------------------
        self.imm = ImageManager()
        self.count = 2
        
        self.kp = 3
        self.kd = 6.2
        self.previous_error = 0
        self.obstacleDetected = False
        self.obstacleAvoidTime = None

    def callback(self, data):
        self.rosImage = data

    def destroy(self):
        cv2.destroyAllWindows()

    def isObstacleAhead(self, front):
        self.obstacleDetected = front < 0.35

    def avoidObstacle(self, front, front_left, left):
        if front < 0.35:
            # print("obstacle ahead, hard turn")
            while self.robot.range[0].distance() > 0.35:
                self.robot.move(0, -0.7)
            self.obstacleAvoidTime = rospy.Time.now()
            return (0.5, 0)
        else:
            current_error = -0.25 + min(front_left, left)

            if(current_error > 1):
                return (0.35, 0.8)
            # The proportional term is the distance from the center of the robot to the line
            proportional_term = self.kp * current_error

            # The derivative term is how quickly the error is changing
            derivative = current_error - self.previous_error
            derivative_term = self.kd * derivative

            output = proportional_term + derivative_term

            self.previous_error = current_error

            return (max(0, 0.75 - abs(output * 1.5)), output)

    def step(self):
        # Leo una imagen del video
        #---------------------------Simulated World-------------------------
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.rosImage, "bgr8")
        except CvBridgeError as e:
            print(e)
        """
        #---------------------------Real World------------------------------
        ret,im = self.capture.read()
        if self.count == 2:
            imNp = im[100:,:]
            #-----------------------------------------------------------------------------------------
            # Normalizamos la imagen con la funcion normalizeImage
            img_norm = self.imm.normalizeImage(imNp)
            # Convertimos la etiqueta en la dimension de la imagen de entrada
            labels = self.imm.predict(img_norm)
            #-----------------------------------------------------------------------------------------
            # Buscamos el contorno de la linea de esa imagen, esto busca si existe una linea
            lineContour = self.imm.findLineContour(labels)
            
            # Detectamos si mediante sensores
            front_all = min([s.distance() for s in self.robot.range["front-all"]])
            front_left = min([s.distance() for s in self.robot.range["left-front"]])
            left = self.robot.range[0].distance()
            
            # Determine if a obstacle is ahead
            if not self.obstacleDetected:
                self.isObstacleAhead(front_all)

            if self.obstacleDetected:
                translation, rotate = self.avoidObstacle(front_all, front_left, left)
                # we do give the robot five seconds to turn around the obstacle until line is founded
                self.obstacleDetected = rospy.Time.now() - self.obstacleAvoidTime < rospy.Duration.from_sec(20.0) or lineContour is None     
                # El robot se intenta recuperar la linea despues de evitar el obstaculo
                if not self.obstacleDetected:
                    self.imm.recoverLineAfterAvoidObstacle(imNp, lineContour)
                    
            elif lineContour is not None:
                #-----------------------------------------------------------------------------------------
                # Analizamos la informacion proporcionada por la imagen de entrada
                cv2.drawContours(imNp, lineContour, -1, (0,255,0),2)
                # Buscamos el contorno distinto de la linea de esa imagen
                notLineContList = self.imm.findNotPartContours(labels,2)
                #-----------------------------------------------------------------------------------------
                # Buscamos el contorno de la marca de esa imagen
                markContour = self.imm.findContours(labels,0)
                # Dibujamos la marca, tambien se analiza si hay flechas
                self.imm.drawContourMark(imNp, labels, markContour, notLineContList)
                #-----------------------------------------------------------------------------------------
                # Obtenemos el error de la linea
                error = self.imm.getLineDetails(imNp, lineContour, notLineContList)
                # Calculos con el error y el sensor
                if(left < 1.0 or front_left < 1.0):
                    rotate = -0.1
                    translation = 0.3
                else:
                    rotate = -error
                    translation = max(0, 1 - abs(error * 1.5)) 
                    if abs(error) < 0.35:
                        translation = translation + 0.15
            else:
                rotate = 0
                translation = -1

            self.robot.move(translation, rotate)
                
            #-----------------------------------------------------------------------------------------
            # Mostrar la imagen procesada
            cv2.imshow("Captura", imNp)
            
            # Convertir la imagen al tipo de datos adecuado antes de escribirla en el archivo de video
            # imNp_8u = cv2.convertScaleAbs(imNp)
            # Escribir la imagen procesada en el archivo de video de salida
            # self.out.write(imNp_8u)
            
            cv2.waitKey(1)

            self.count = 0
        else:
            self.count  = self.count + 1
    
def INIT(engine):
    assert (engine.robot.requires("range-sensor") and
            engine.robot.requires("continuous-movement"))

    return BrainFollowLine('BrainFollowLine', engine)

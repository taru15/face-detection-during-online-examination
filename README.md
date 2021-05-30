# face-detection-during-online-examination
Table of Contents
1.	Objective (Problem Statement)
2.	Requirement Analysis (SRS)
2.1.	Requirement specification
                   2.1.1    Performance Requirements
                   2.1.2    Safety Requirements 
                   2.1.3    Security Requirements 
2.2.	H/w and S/w Requirements
2.3.	Feasibility Study
                   2.3.1     Technical
                   2.3.2     Portability
                   2.3.3     Adaptability
2.4.	Product Functions
2.5.	Use-case Diagrams
3.	System Design (SDS)
3.1.	ER Diagram/Class Diagrams
3.2.	Data flow diagrams/Activity Diagrams
3.3.	Flowcharts/Sequence Diagrams
4.	Coding 
4.1.	Only main Modules 
5.	User Interfaces
6.	References
 
1. Objective
The Objective of this project is to develop a Face Detection system to identify or verify the identity of an individual using their face. Our main focus is towards the face detection during an online examination system. It can be used in conducting cheating free examinations as the movements of the students are captured by the web-cam. This application can have tremendous scope of face detection. This system can be effectively used in ATM’s, identifying duplicate voters, passport and visa verification, driving license verification, in defence, competitive and other exams, in governments and private sectors.

2. Requirement Analysis (SRS)

      2.1Requirement specification
2.1.1 Performance Requirements
This system performs well when users login for examination and then            start examination. To conduct the exam smoothly users should not try to cheat or should not be any kind of unfair.   
2.1.2 Safety Requirements 
There is no safety requirement for this  application.
2.1.3 Security Requirements 
There is no specific security requirement.


2.2 H/w and S/w Requirements
  Hardware Requirements
i.	4 GB RAM (Minimum)
ii.	80 GB HDD
iii.	Dual Core processor
iv.	CDROM (installation only). VGA resolution monitor
v.	Microsoft Windows 98/2000/NT with service pack 6 / XP with service pack 2/ Windows 7 with service pack 2
vi.	SQL Server 2008 R2
vii.	OpenCV-python 4.2.0.32
Software Requirements
i.	OpenCV-python 4.2.0.32: OpenCV-Python is a library of Python bindings designed to solve computer vision problems.
ii.	MS-excel 
iii.	Scripting Language: Python is an interpreted, high-level and general-purpose programming language.
iv.	Operating system: Windows 7, 8, 10, 64-bit versions only; macOS 10.12+, Android version above 7.0 and Visual Studio are supported.
v.	Libraries of python: pillow, dlib, thinter, numpy, cv2, os, shutil and pandas.
 
2.3 Feasibility Study
2.3.1 Technical
The project is feasible through the use of a computer which is the hardware. The app will not crash or hang except as the result of operating system error. The software involved includes: python and opencv libraries. The project is going to have a positive impact socially as it will minimize cheating while online examination.
2.3.2 Portability
Portability not available on any other platform except windows.
2.3.3 Adaptability
o	Any OS running on the system shall be able to run the application. 
o	Users must have a web-camera to use this application.
2.4 Product Functions
The application will detect and recognize student faces if the face dataset in the database has the same lighting and resolution when capturing images. When the face is recognized by the application, the application will confirm whether the face that is recognized by the system is the face of the student concerned. If students confirm correctly, the exam questions will come out and students take the exam. The second stage, the process of inputting face data on the system can be seen in the first user selecting the input method, that is, from existing photo files or directly via webcam, then it will be displayed on the Image Box Webcam. The system will detect faces. In the trials that have been carried out the camera detects changes in facial expression as well as body movements.



2.5 Use-case Diagrams








3.System Design (SDS)

    3.1 Class Diagrams
 
3.2 Data flow diagrams
Backend Process




























Yes
Yes			        Yes			    Yes			          


Frontend Process




















 


3.3 Sequence Diagrams

4. Coding
  4.1 Main code:-
    import read
    import Train
    import face_login
    import cheating
    # to get the details of student and reading the face
    read.TakeImages()
   # to training the data into a yml file
    Train.TrainImages()
    #detecting the face for login purpose
    face_login.DetectFace()

    print('start your exam')
    #to mark any unfair mean
    cheating.DetectFace()

 5. User Interfaces










 


6. References
1.	R. Bhatia, “Biometrics and Face Recognition Techniques,” Int. J. Adv. Res. Comput. Sci. Softw. Eng., vol. 3, no. 5, pp. 93–99, 2013.			
2.	S. Syed Navaz, T. D. Sri, P. Mazumder, and A. Professor, “Face Recognition Using Principal Component Analysis and Neural Networks,” Int. J. Comput. Networking, Wirel. Mob. Commun., vol. 3, no. 1, pp. 2250–1568, 2013.
3.	M. K. Dhavalsinh V. Solanki, “A Survey on Face Recognition Techniques,” J. Image Process. Pattern Recognit. Prog., vol. 4, no. 6, pp. 11–16, 2013.
4.	https://www.slideshare.net/derekbudde/face-detection-and-recognition
5.	https://towardsdatascience.com/how-to-build-a-face-detection-and-recognition-system-f5c2cdfbeb8c


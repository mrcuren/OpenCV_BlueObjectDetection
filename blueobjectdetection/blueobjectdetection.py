
import numpy as np
import cv2 
import pandas as pd
import time
"""
Kütüphanelerimizi projemize dahil ediyoruz.
"""
sure1=time.time()
prev_frame_time = 0
new_frame_time = 0

kamera = cv2.VideoCapture(0) #cv2.VİDEOcapture fonksiyonu ile bilgisayara bağlı olan kameranın ID si girilerek o kameradan görüntülerimizi aldık.


while True:
    _,kare = kamera.read() #Alınan görüntüyü okuyoruz.
    new_frame_time = time.time()
    
    cv2.imshow("Kamera",kare)#Kamerayı ekrana yansıtıyoruz.
    blur = cv2.GaussianBlur(kare,(7,7),5) #Gürültülerden arınmak için blurlama işlemini gerçekleştiriyoruz.
    #Nesneyi daha ön planda tutması için cv2.GaussianBlur methodunu tercih ettim.
    hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV) #Görüntümüzü ,RGB renk kanalında eşikleme yapılmadığından ve HSV renk skalasının daha kullanışlı olması açısından HSVye dönüştürüyoruz.
    lower = np.array([75,80,80])   #Mavi değerlerin bulunduğu alt limit değerlerini belirliyoruz.
    upper = np.array([130,255,255]) #Mavi değerlerin bulunduğu üst limit değerlerini belirliyoruz.
    maske = cv2.inRange(hsv,lower,upper) # cv2.inRange methoduyla mavi renkte olan piksel değerlerimizi maskeliyoruz.
    cv2.imshow("Maske",maske)    #Maskelenmiş yani threshold yaptığımız resmi ekrana yansıtıyoruz.
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
 
    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)
 
    # putting the FPS count on the frame
    cv2.putText(kare, fps, (7, 70), cv2.FONT_HERSHEY_PLAIN, 3, (100, 255, 0), 3, cv2.LINE_AA)

    
    contours, _ = cv2.findContours(maske, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #findContours methodu sayesinde maskelenmiş görüntümüze kontur işlemi gerçekleştiriyoruz.
    if len(contours)>0: #kontur değerlerin 0'dan yüksek olup olmadığını kontrol ediyoruz bu sayede kontur var ise işlemleri gerçekleştireceğiz.
        area_list = [] #Alanları alacağımız boş liste oluşturuyoruz.
        for cnt in contours:
            area = cv2.contourArea(cnt) #Kontur alanlarını area değişkenine atıyoruz.
            area_list.append(area) #listemize ekliyoruz.
            epsilon = 0.01*cv2.arcLength(cnt, True) #arcLength fonksiyonuyla kontur çevresininin uzunluğunu hesaplıyoruz. 
            approx = cv2.approxPolyDP(cnt, epsilon, True) #approxPolyDP methodu tamamlanmamış eğriyi tahmin etmemize yardımcı olur.
            #Örneğin kare mavi rengin üzerinde elimiz var ise burada kopukluk yaşanacaktır ancak fonksiyonumuz bunu tekrardan "uydurarak" tamamlamamızı sağlar. 
            cv2.drawContours(kare, [approx], 0, (0,255,255), 3) #konturumuzu çizdiriyoruz.
            #cv2.imshow("approxPolyDP",kare)
            
                
            
            
            
        index = area_list.index(max(area_list)) #Alan listemizde yer alan en büyük değerimizin index numarasını index değişkenine alıyoruz.
        rect = cv2.minAreaRect(contours[index]) #Kontur değerlerimizdeki en büyük alana sahip konturlerin minimum alanla çizdirme işlemini yapıyoruz.
        (x,y), (w,h), rotation = rect #minAreaRect fonksiyonu bizlere konumumuz ve genişlik-uzunluk hakkında bilgi verir. Verdiği bilgileri değişkenlere kaydediyoruz.
        string = "x:{}, y:{}, w:{}, h:{}, rotation:{}".format(np.round(x),np.round(y),np.round(w),np.round(h),np.round(rotation)) #verilen anlık bilgiler ekrana yazdırılmak için metin haline hazırlanır.
        cv2.putText(kare,string,(250,460),cv2.FONT_HERSHEY_PLAIN,1,[0,0,0],1,cv2.LINE_AA) #ekranın sağ alt kısmına yazdırılır.
        alan=w*h #♦genişlik ve uzunluk çarpılarak alanı buluruz.
        string = "Tahmini Alan:{}".format(np.round(alan)) #alan yazdırılmak üzere metin halinde hazırlanır.
        cv2.putText(kare,string,(10,40),cv2.FONT_HERSHEY_PLAIN,1,[0,0,0],1,cv2.LINE_AA)#ekranın sol üst kısmına alan bilgisi yazdırılır.
        box = cv2.boxPoints(rect) #verilen konumlar üzerinden 4 köşe tespit edilir ve etrafı çizdirilmek istenir.
        box = np.int64(box)#tespit edilen 4 köşe integer formatına dönüştürülür .
        cv2.drawContours(kare,[box],0,[0,0,255],thickness=3) #Tespit edilen bölgeyi 4 köşeli bir şekilde ekranda belirtiriz.
        
        if len(approx) == 3: #Eğer 3 köşe algılandıysa mavi renkli nesnemize üçgen diyebiliriz.
            adi = "Algilanan nesne: Ucgen "
            cv2.putText(kare,adi,(10,20),cv2.FONT_HERSHEY_PLAIN,1,[0,0,0],1,cv2.LINE_AA)
        elif len(approx) == 4:#Eğer 4 köşe algılandıysa mavi renkli nesnemize dikdörtgen veya kare diyebiliriz.
            if abs(w-h) <= 3: #Eğer Genişliğin uzunluktan farkı 3 ve 3'den az ise nesnemiz kare
                adi="Algilanan nesne: Kare"
                cv2.putText(kare,adi,(10,20),cv2.FONT_HERSHEY_PLAIN,1,[0,0,0],1,cv2.LINE_AA)
            else :#Eğer Genişliğin uzunluktan farkı 3'den fazla ise nesnemiz dikdörtgen olabilir diyebiliriz.
                adi = "Algilanan nesne: Dikdortgen"
                cv2.putText(kare,adi,(10,20),cv2.FONT_HERSHEY_PLAIN,1,[0,0,0],1,cv2.LINE_AA)
        elif len(approx) >=15:  #Eğer 15'den fazla köşe algılandıysa mavi renkli nesnemize daire diyebiliriz. Genelde 15 rakamı kullanıldığı için bunu tercih ettim.
            adi = "Algilanan nesne: Daire"
            cv2.putText(kare,adi,(10,20),cv2.FONT_HERSHEY_PLAIN,1,[0,0,0],1,cv2.LINE_AA)
            
        
        
        
        C = cv2.moments(contours[index]) #Merkez değerini hesaplayabilmek için cv2.moments fonksiyonundan yararlanıyoruz.
        #Kontur değerlerimizdeki en büyük alana sahip konturlerin anlarını buluyoruz.
        #Hatayı(ZeroDivisionError) kontrol edebilmemiz için try except bloklarını kullanıyoruz. 
        try:
            #Merkezin x ve y koordinatlarını hesaplıyoruz.
            x = int(C["m10"]/C["m00"])
            y = int(C["m01"]/C["m00"])
            center = (x,y)
            cv2.circle(kare,center,3,[0,255,0],-1) #Görüntümüzün merkezine daire çizdiriyoruz.
        except :
            pass

        cv2.imshow("Mavi Renk Filtresi Sonucumuz",kare) #Sonucumuzu ekrana yansıtıyoruz.
        

    key = cv2.waitKey(25) #Her bir saniyede ekrana 40 kare gelir.
    if key == 27: #Eğer ESC tuşuna basılırsa programı sonlandır.
        break

kamera.release() #Kamerayı rahat bırak.
cv2.destroyAllWindows() # Ram üzerindeki Bütün penceleri sonlandır.
sure2=time.time()
print(sure2-sure1)

import time
import sys
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from configparser import ConfigParser
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from configparser import ConfigParser
from selenium.webdriver.chrome.options import Options
from datetime import datetime,date,timedelta 
import getpass
username = getpass.getuser()
import os
import smtplib
from string import Template
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders 
from email.mime.base import MIMEBase

class TableauReportExtraction:
    
    def __init__(self):
        print ("TableauReportExtraction Init called")
        self.parser = ConfigParser(interpolation=None)
        self.parser.read('./config1.ini')
        self.username = self.parser.get('USER1', 'username')
        self.user = self.parser.get('USER1', 'user')
        self.password = self.parser.get('USER1', 'password')
        self.report_url = self.parser.get('USER1', 'report_url')
        
        self.extension = self.parser.get('USER1', 'b_nav')
        self.driver = r"{}".format(self.parser.get('Drivers', 'chrome_ver87'))
        self.view_name = self.parser.get('USER1', 'view_name')
        print(self.driver)
    
        self.dst_path = 'C:\\Users\\'+username+'\\Tableau Exports\\'+self.user+'\\'
        self.str_date = datetime.now().strftime('%Y%m%d')
        global dst, my_mail_add, pwd
        dst = self.dst_path+self.str_date
        my_mail_add = self.parser.get('Mailing_Address', 'email')
        pwd = self.parser.get('Mailing_Address', 'pwd')
#         username = getpass.getuser()
        

    def loginDetails(self,driver):
        
        driver.get(self.report_url)
        print("Report URL has been launched")
        
        
        try:
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="ng-app"]/div/div[1]/span/div[2]/span/div/div[2]/span/form/div[1]/div[1]/div/div/input')))
            
        except:
            print('The loading of the page take more than a usual time')
        
        try:
            username_elem = driver.find_element_by_xpath('//*[@id="ng-app"]/div/div[1]/span/div[2]/span/div/div[2]/span/form/div[1]/div[1]/div/div/input')
            password_elem = driver.find_element_by_xpath('//*[@id="ng-app"]/div/div[1]/span/div[2]/span/div/div[2]/span/form/div[1]/div[2]/div/div/input')
            login = driver.find_element_by_xpath('//*[@id="ng-app"]/div/div[1]/span/div[2]/span/div/div[2]/span/form/button/span[1]/span[1]')
            username_elem.send_keys(self.username)
            password_elem.send_keys(self.password)
            login.click()

            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, '//*[@id="viz"]/iframe')))
            iframe=driver.find_element_by_xpath('//*[@id="viz"]/iframe')
            driver.switch_to.frame(iframe)
   
        except:
            print('Got some exception while loging to the Tableau Server\nCheck the field or the User Account Details')
    
    def promptSelection(self,driver):
  
            
        try:
            time.sleep(5)
                
            driver.find_element_by_xpath('//*[@id="extensions-permissions-dialog"]/div[2]/div[2]/button').click()
            print("Closed Dialog Box Successfuly")
            
        except:
            print("No dialog box")

        time.sleep(200)
        iframe=driver.find_element_by_xpath('//*[@id="extension_frame_298"]')

        driver.switch_to.frame(iframe)
        
        time.sleep(25)
            
        ext = driver.find_element_by_xpath(self.extension)
        ext.click()
        
    def get_contacts(self,filename):
        names = []
        emails = []
        with open(filename, mode='r', encoding='utf-8') as contacts_file:
            for a_contact in contacts_file:
                names.append(a_contact.split()[0])
                emails.append(a_contact.split()[1])
        return names, emails


    def read_template(self,filename):
        with open(filename, 'r', encoding='utf-8') as template_file:
            template_file_content = template_file.read()
        return Template(template_file_content)

    def send_mail(self):
        names, emails = self.get_contacts('mycontacts.txt')
        message_template = self.read_template('message.txt')

        # setting up the SMTP server
        s = smtplib.SMTP(host='smtp-mail.outlook.com', port=587)
        s.starttls()
        s.login(my_mail_add, pwd)

        # For each contact, send the email:
        for name, email in zip(names, emails):
            msg = MIMEMultipart()       # create a message

            # add in the actual person name to the message template
            message = message_template.substitute(PERSON_NAME=name.title())

            # Prints out the message body for our sake
            print(message)

            # setup the parameters of the message
            msg['From']=my_mail_add
            msg['To']=email
            msg['Subject']="Report Bursting PoC"

            # add in the message body
            msg.attach(MIMEText(message, 'plain'))


            # open the file to be sent  
            filename1 = self.view_name+'.xlsx'

            attachment = open(dst, "rb")
            #attachment = open('C:\\Users\\'+username, "rb") 

            # instance of MIMEBase and named as p 
            p = MIMEBase('application', 'octet-stream') 

            # To change the payload into encoded form 
            p.set_payload((attachment).read()) 

            # encode into base64 
            encoders.encode_base64(p) 

            p.add_header('Content-Disposition', "attachment; filename= %s" % filename1) 

            # attach the instance 'p' to instance 'msg' 
            msg.attach(p)
            # send the message via the server set up earlier.
            s.send_message(msg)
            del msg

        # Terminate the SMTP session and close the connection
        s.quit()
    
    
        
    def sequenceExceutor(self):
        
        #creating driver
        
        print(dst)
        if not os.path.exists(self.dst_path):
            os.makedirs(self.dst_path)
            if not os.path.exists(self.dst_path+self.str_date):
                os.makedirs(self.dst_path+self.str_date)
        options = Options()
        options.add_experimental_option("prefs", {
          "download.default_directory":(self.dst_path+self.str_date),
          "download.prompt_for_download": False,
          "download.directory_upgrade": True,
          "safebrowsing.enabled": False
        })
        driver = webdriver.Chrome(executable_path= self.driver,options=options)
        driver.maximize_window()

        #call loginDetails
        self.loginDetails(driver)

        #call promptSelection
        self.promptSelection(driver)
        
        self.send_mail()

object = TableauReportExtraction()
object.sequenceExceutor()

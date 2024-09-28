# from flask import Flask, render_template, request
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

def scrape_doctors(specialist, location):
    # Set up Selenium WebDriver options for headless mode
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Enable headless mode
    chrome_options.add_argument("--disable-gpu")  # Disable GPU hardware acceleration
    chrome_options.add_argument("--no-sandbox")  # Bypass OS security model
    chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems
    
    # Initialize the WebDriver with the options
    driver = webdriver.Chrome(options=chrome_options)

    try:
        # Construct the URL dynamically based on the specialization and location
        url = f"https://www.practo.com/search/doctors?results_type=doctor&q=%5B%7B%22word%22%3A%22{specialist}%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22subspeciality%22%7D%5D&city={location}"
        driver.get(url)
        
        # Wait for page to load completely
        time.sleep(5)

        # Scroll down to load more doctors (if necessary)
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
        time.sleep(2)  # Wait for the page to load more content

        # Parse the page with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Find all doctor sections
        doctor_sections = soup.find_all('div', class_='info-section')

        # Initialize an empty list to store doctors' information
        doctors = []

        # Loop through each doctor section and extract details
        for index, section in enumerate(doctor_sections):
            # print(f"Processing section {index + 1}")

            # Extract doctor's name
            doctor_name_tag = section.find('h2', class_='doctor-name')
            doctor_name = doctor_name_tag.text.strip() if doctor_name_tag else "Not available"
            
            # Extract specialization
            specialization_tag = section.find('div', class_='u-d-flex')
            specialization = specialization_tag.text.strip() if specialization_tag else "Not available"
            
            # Extract experience
            experience_tag = section.find('div', {'data-qa-id': "doctor_experience"})
            experience = experience_tag.text.strip().split()[0] if experience_tag else "Not available"
            
            # Extract clinic locality and city
            locality_tag = section.find('span', {'data-qa-id': 'practice_locality'})
            city_tag = section.find('span', {'data-qa-id': 'practice_city'})
            location = f"{locality_tag.text.strip()}, {city_tag.text.strip()}" if locality_tag and city_tag else "Location not available"
            
            # Extract clinic name
            clinic_name_tag = section.find('span', {'data-qa-id': 'doctor_clinic_name'})
            clinic_name = clinic_name_tag.text.strip() if clinic_name_tag else "Clinic name not available"
            
            # Extract consultation fee
            fee_tag = section.find('span', {'data-qa-id': 'consultation_fee'})
            consultation_fee = fee_tag.text.strip() if fee_tag else "Fee not available"
            
            # Append doctor details to the list as a dictionary
            doctor_info = {
                "name": doctor_name,
                "specialization": specialization,
                "experience": experience + " years",
                "location": location,
                "clinic_name": clinic_name,
                "consultation_fee": consultation_fee
            }

            
            doctors.append(doctor_info)

        return doctors
    
      # Move the return outside the loop to return all data
    except Exception as e:
        print(f"An error occurred: {e}")
        return [] 
    finally:
        # Ensure the driver is closed
        driver.quit()

# Example usage
# print(scrape_doctors('Dentist', 'Mumbai'))

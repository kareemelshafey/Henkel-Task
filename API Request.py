import json
import requests

# Set the URL of the API endpoint
url = "http://localhost:8000/predict"

# Define the feature values
feature_values = [1.0, 2.0, 3.0]

# Create the request payload as a JSON object
payload = json.dumps({"values": feature_values})

# Set the content type header to indicate JSON data
headers = {"Content-Type": "application/json"}

try:
    # Send the POST request to the API endpoint
    response = requests.post(url, data=payload, headers=headers, timeout=1)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Print the response
        print(response.json())
    else:
        # Raise an exception with the error message from the response
        response_json = response.json()
        error_message = response_json.get('detail', 'Unknown error')
        raise Exception(f"Request failed with status code {response.status_code}: {error_message}")

except Exception as e:
    # Catch any exception that may occur
    raise Exception(f"Error occurred: {str(e)}")






# api/response.py
"""
Standardized response formatting for the API.
"""

from flask import jsonify

def success_response(message, data=None):
    """
    Generate a standardized success response
    """
    response = {
        "status": "success",
        "message": message
    }
    
    if data is not None:
        response["data"] = data
    
    return jsonify(response)

def error_response(message, status_code=400):
    """
    Generate a standardized error response
    """
    response = {
        "status": "error",
        "message": message
    }
    
    return jsonify(response), status_code
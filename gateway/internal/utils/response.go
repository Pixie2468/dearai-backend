package utils

import (
	"encoding/json"
	"net/http"
)

// ErrorResponse represents a standardized API error shape.
type ErrorResponse struct {
	Error string `json:"error"`
}

// RespondJSON marshals a payload to JSON and writes the HTTP response.
func RespondJSON(w http.ResponseWriter, status int, payload any) {
	response, err := json.Marshal(payload)
	if err != nil {
		// Log the actual error internally here in a real app (e.g., log.Printf)

		// 1. ALWAYS set headers before WriteHeader
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)

		// 2. Fixed the malformed JSON string syntax
		w.Write([]byte(`{"error": "internal server error: failed to serialize response"}`))
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	w.Write(response)
}

// RespondError sends a standardized JSON error message.
func RespondError(w http.ResponseWriter, status int, message string) {
	RespondJSON(w, status, ErrorResponse{Error: message})
}

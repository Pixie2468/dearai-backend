package utils

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestRespondJSONSuccess(t *testing.T) {
	recorder := httptest.NewRecorder()

	RespondJSON(recorder, http.StatusCreated, map[string]string{"ok": "true"})

	if recorder.Code != http.StatusCreated {
		t.Fatalf("expected status %d, got %d", http.StatusCreated, recorder.Code)
	}

	if contentType := recorder.Header().Get("Content-Type"); contentType != "application/json" {
		t.Fatalf("expected Content-Type application/json, got %q", contentType)
	}

	var payload map[string]string
	if err := json.Unmarshal(recorder.Body.Bytes(), &payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if payload["ok"] != "true" {
		t.Fatalf("expected ok true, got %q", payload["ok"])
	}
}

func TestRespondError(t *testing.T) {
	recorder := httptest.NewRecorder()

	RespondError(recorder, http.StatusBadRequest, "bad request")

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("expected status %d, got %d", http.StatusBadRequest, recorder.Code)
	}

	var payload ErrorResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if payload.Error != "bad request" {
		t.Fatalf("expected error message, got %q", payload.Error)
	}
}

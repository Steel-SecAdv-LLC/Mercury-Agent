{{/*
Mercury-Agent Helm Chart Helper Templates
*/}}

{{/*
Expand the name of the chart.
*/}}
{{- define "mercury-agent.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
*/}}
{{- define "mercury-agent.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "mercury-agent.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "mercury-agent.labels" -}}
helm.sh/chart: {{ include "mercury-agent.chart" . }}
{{ include "mercury-agent.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- with .Values.commonLabels }}
{{ toYaml . }}
{{- end }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "mercury-agent.selectorLabels" -}}
app.kubernetes.io/name: {{ include "mercury-agent.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
API component labels
*/}}
{{- define "mercury-agent.api.labels" -}}
{{ include "mercury-agent.labels" . }}
app.kubernetes.io/component: api
{{- end }}

{{/*
API selector labels
*/}}
{{- define "mercury-agent.api.selectorLabels" -}}
{{ include "mercury-agent.selectorLabels" . }}
app.kubernetes.io/component: api
{{- end }}

{{/*
Engine component labels
*/}}
{{- define "mercury-agent.engine.labels" -}}
{{ include "mercury-agent.labels" . }}
app.kubernetes.io/component: engine
{{- end }}

{{/*
Engine selector labels
*/}}
{{- define "mercury-agent.engine.selectorLabels" -}}
{{ include "mercury-agent.selectorLabels" . }}
app.kubernetes.io/component: engine
{{- end }}

{{/*
Create the name of the API service account to use
*/}}
{{- define "mercury-agent.api.serviceAccountName" -}}
{{- if .Values.api.serviceAccount.create }}
{{- default (printf "%s-api" (include "mercury-agent.fullname" .)) .Values.api.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.api.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the name of the Engine service account to use
*/}}
{{- define "mercury-agent.engine.serviceAccountName" -}}
{{- if .Values.engine.serviceAccount.create }}
{{- default (printf "%s-engine" (include "mercury-agent.fullname" .)) .Values.engine.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.engine.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the image name
*/}}
{{- define "mercury-agent.image" -}}
{{- $tag := default .Chart.AppVersion .Values.api.image.tag }}
{{- printf "%s:%s" .Values.api.image.repository $tag }}
{{- end }}

{{/*
Create the engine image name
*/}}
{{- define "mercury-agent.engine.image" -}}
{{- $tag := default .Chart.AppVersion .Values.engine.image.tag }}
{{- printf "%s:%s" .Values.engine.image.repository $tag }}
{{- end }}

{{/*
Return the namespace
*/}}
{{- define "mercury-agent.namespace" -}}
{{- if .Values.namespace.create }}
{{- .Values.namespace.name | default .Release.Namespace }}
{{- else }}
{{- .Release.Namespace }}
{{- end }}
{{- end }}

{{/*
Return the ConfigMap name
*/}}
{{- define "mercury-agent.configMapName" -}}
{{- printf "%s-config" (include "mercury-agent.fullname" .) }}
{{- end }}

{{/*
Return the Secret name
*/}}
{{- define "mercury-agent.secretName" -}}
{{- if .Values.config.secrets.existingSecret }}
{{- .Values.config.secrets.existingSecret }}
{{- else }}
{{- printf "%s-secrets" (include "mercury-agent.fullname" .) }}
{{- end }}
{{- end }}

{{/*
Return the PVC names
*/}}
{{- define "mercury-agent.dataPvcName" -}}
{{- if .Values.persistence.data.existingClaim }}
{{- .Values.persistence.data.existingClaim }}
{{- else }}
{{- printf "%s-data" (include "mercury-agent.fullname" .) }}
{{- end }}
{{- end }}

{{- define "mercury-agent.modelsPvcName" -}}
{{- if .Values.persistence.models.existingClaim }}
{{- .Values.persistence.models.existingClaim }}
{{- else }}
{{- printf "%s-models" (include "mercury-agent.fullname" .) }}
{{- end }}
{{- end }}

{{/*
Checksum for ConfigMap to trigger pod restarts on config changes
*/}}
{{- define "mercury-agent.configChecksum" -}}
{{- include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
{{- end }}

{{/*
Checksum for Secret to trigger pod restarts on secret changes
*/}}
{{- define "mercury-agent.secretChecksum" -}}
{{- include (print $.Template.BasePath "/secret.yaml") . | sha256sum }}
{{- end }}

{{/*
Create the annotations for pods
*/}}
{{- define "mercury-agent.podAnnotations" -}}
checksum/config: {{ include "mercury-agent.configChecksum" . }}
checksum/secret: {{ include "mercury-agent.secretChecksum" . }}
{{- with .Values.podAnnotations }}
{{ toYaml . }}
{{- end }}
{{- if .Values.metrics.enabled }}
prometheus.io/scrape: "true"
prometheus.io/port: {{ .Values.metrics.port | quote }}
prometheus.io/path: {{ .Values.metrics.path | quote }}
{{- end }}
{{- end }}

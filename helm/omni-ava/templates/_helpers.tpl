{{/*
OMNI-AVA Helm Chart Helper Templates
*/}}

{{/*
Expand the name of the chart.
*/}}
{{- define "omni-ava.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
*/}}
{{- define "omni-ava.fullname" -}}
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
{{- define "omni-ava.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "omni-ava.labels" -}}
helm.sh/chart: {{ include "omni-ava.chart" . }}
{{ include "omni-ava.selectorLabels" . }}
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
{{- define "omni-ava.selectorLabels" -}}
app.kubernetes.io/name: {{ include "omni-ava.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
API component labels
*/}}
{{- define "omni-ava.api.labels" -}}
{{ include "omni-ava.labels" . }}
app.kubernetes.io/component: api
{{- end }}

{{/*
API selector labels
*/}}
{{- define "omni-ava.api.selectorLabels" -}}
{{ include "omni-ava.selectorLabels" . }}
app.kubernetes.io/component: api
{{- end }}

{{/*
Engine component labels
*/}}
{{- define "omni-ava.engine.labels" -}}
{{ include "omni-ava.labels" . }}
app.kubernetes.io/component: engine
{{- end }}

{{/*
Engine selector labels
*/}}
{{- define "omni-ava.engine.selectorLabels" -}}
{{ include "omni-ava.selectorLabels" . }}
app.kubernetes.io/component: engine
{{- end }}

{{/*
Create the name of the API service account to use
*/}}
{{- define "omni-ava.api.serviceAccountName" -}}
{{- if .Values.api.serviceAccount.create }}
{{- default (printf "%s-api" (include "omni-ava.fullname" .)) .Values.api.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.api.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the name of the Engine service account to use
*/}}
{{- define "omni-ava.engine.serviceAccountName" -}}
{{- if .Values.engine.serviceAccount.create }}
{{- default (printf "%s-engine" (include "omni-ava.fullname" .)) .Values.engine.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.engine.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the image name
*/}}
{{- define "omni-ava.image" -}}
{{- $tag := default .Chart.AppVersion .Values.api.image.tag }}
{{- printf "%s:%s" .Values.api.image.repository $tag }}
{{- end }}

{{/*
Create the engine image name
*/}}
{{- define "omni-ava.engine.image" -}}
{{- $tag := default .Chart.AppVersion .Values.engine.image.tag }}
{{- printf "%s:%s" .Values.engine.image.repository $tag }}
{{- end }}

{{/*
Return the namespace
*/}}
{{- define "omni-ava.namespace" -}}
{{- if .Values.namespace.create }}
{{- .Values.namespace.name | default .Release.Namespace }}
{{- else }}
{{- .Release.Namespace }}
{{- end }}
{{- end }}

{{/*
Return the ConfigMap name
*/}}
{{- define "omni-ava.configMapName" -}}
{{- printf "%s-config" (include "omni-ava.fullname" .) }}
{{- end }}

{{/*
Return the Secret name
*/}}
{{- define "omni-ava.secretName" -}}
{{- if .Values.config.secrets.existingSecret }}
{{- .Values.config.secrets.existingSecret }}
{{- else }}
{{- printf "%s-secrets" (include "omni-ava.fullname" .) }}
{{- end }}
{{- end }}

{{/*
Return the PVC names
*/}}
{{- define "omni-ava.dataPvcName" -}}
{{- if .Values.persistence.data.existingClaim }}
{{- .Values.persistence.data.existingClaim }}
{{- else }}
{{- printf "%s-data" (include "omni-ava.fullname" .) }}
{{- end }}
{{- end }}

{{- define "omni-ava.modelsPvcName" -}}
{{- if .Values.persistence.models.existingClaim }}
{{- .Values.persistence.models.existingClaim }}
{{- else }}
{{- printf "%s-models" (include "omni-ava.fullname" .) }}
{{- end }}
{{- end }}

{{/*
Checksum for ConfigMap to trigger pod restarts on config changes
*/}}
{{- define "omni-ava.configChecksum" -}}
{{- include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
{{- end }}

{{/*
Checksum for Secret to trigger pod restarts on secret changes
*/}}
{{- define "omni-ava.secretChecksum" -}}
{{- include (print $.Template.BasePath "/secret.yaml") . | sha256sum }}
{{- end }}

{{/*
Create the annotations for pods
*/}}
{{- define "omni-ava.podAnnotations" -}}
checksum/config: {{ include "omni-ava.configChecksum" . }}
checksum/secret: {{ include "omni-ava.secretChecksum" . }}
{{- with .Values.podAnnotations }}
{{ toYaml . }}
{{- end }}
{{- if .Values.metrics.enabled }}
prometheus.io/scrape: "true"
prometheus.io/port: {{ .Values.metrics.port | quote }}
prometheus.io/path: {{ .Values.metrics.path | quote }}
{{- end }}
{{- end }}

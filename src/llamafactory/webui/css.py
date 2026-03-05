# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

CSS = r"""
.duplicate-button {
  margin: auto !important;
  color: white !important;
  background: black !important;
  border-radius: 100vh !important;
}

.thinking-summary {
  padding: 8px !important;
}

.thinking-summary span {
  border-radius: 4px !important;
  padding: 4px !important;
  cursor: pointer !important;
  font-size: 14px !important;
  background: rgb(245, 245, 245) !important;
}

.dark .thinking-summary span {
  background: rgb(73, 73, 73) !important;
}

.thinking-container {
  border-left: 2px solid #a6a6a6 !important;
  padding-left: 10px !important;
  margin: 4px 0 !important;
}

.thinking-container p {
  color: #a6a6a6 !important;
}

.modal-box {
  position: fixed !important;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%); /* center horizontally */
  max-width: 1000px;
  max-height: 750px;
  overflow-y: auto;
  background-color: var(--input-background-fill);
  flex-wrap: nowrap !important;
  border: 2px solid black !important;
  z-index: 1000;
  padding: 10px;
}

.dark .modal-box {
  border: 2px solid white !important;
}

.config-preview-container {
  position: relative;
  overflow: visible;
  z-index: 9998;
}

.config-preview-container .gradio-row,
.config-preview-container .gradio-column,
.config-preview-container .form {
  overflow: visible !important;
}

.config-preview-container .block,
.config-preview-container .wrap,
.config-preview-container .form {
  overflow: visible !important;
}

.config-preview {
  opacity: 0;
  visibility: hidden;
  position: absolute;
  top: calc(100% + 6px);
  left: 0;
  z-index: 9999;
  min-width: 360px;
  max-width: 720px;
  max-height: 480px;
  overflow: auto;
  padding: 10px;
  border: 1px solid #a6a6a6;
  background: var(--input-background-fill);
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
  transition: opacity 0.12s ease-in-out;
  transition-delay: 0.6s;
}

.config-preview p,
.config-preview li,
.config-preview pre {
  white-space: pre-wrap;
}

.config-preview-container:hover .config-preview,
.config-preview-container:focus-within .config-preview {
  opacity: 1;
  visibility: visible;
  transition-delay: 0.6s;
}

.btn-start button {
  background: #2e7d32 !important;
  border-color: #2e7d32 !important;
  color: white !important;
}

.btn-stop button {
  background: #f57c00 !important;
  border-color: #f57c00 !important;
  color: white !important;
}
"""


Google Drive Tools
==================
This is a first attempt to develop tools to read and write to Google Drive.  

This script is configured with an [OAuth2 client ID and secret for desktop apps][OAuth for desktop apps].  You can create the necessary credentials [here][credentials], and save them as `credentials.json`.  

The script runs with [OAuth Consent Flow][OAuth Consent Flow] to execute with the authorisation of a single user, and will open a browser window to prompt the user for permission the first time the script executes.  The returned access token (and scope) are automatically saved as `token.json`

[OAuth for desktop apps]:https://developers.google.com/identity/protocols/oauth2/native-app
[credentials]:https://console.cloud.google.com/apis/credentials
[OAuth Consent Flow]:https://cloud.google.com/docs/authentication/end-user
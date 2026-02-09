## Deploy an Agent
To deploy an Agent Engine app using adk deploy agent_engine, complete the following steps:

1. In the adk_to_agent_engine/transcript_summarization_agent directory, click on the agent.py file to review the instructions of this simple summarization agent.
2. In the Cloud Shell Terminal, run the deploy command:
adk deploy agent_engine transcript_summarization_agent \
--display_name "Transcript Summarizer" \
--region us-central1 \
--staging_bucket gs://qwiklabs-gcp-00-7f231bacc2f8-bucket

## Query a deployed Agent
To query the agent, you must first grant it the authorization to call models via Vertex AI.

1. To see the service agent and its assigned role, navigate to IAM in the console.
2. Click the checkbox to Include Google-provided role grants.
3. Find the AI Platform Reasoning Engine Service Agent (service-PROJECT_NUMBER@gcp-sa-aiplatform-re.iam.gserviceaccount.com), and click the edit pencil icon in this service agent's row.
4. Click + Add another role.
5. In the Select a role field, enter Vertex AI User. If you deploy an agent that uses tools to access other data, you would grant access to those systems to this service agent as well.
6. Save your changes.
7. In the Cloud Shell Terminal, run the file from the adk_to_agent_engine directory with:
cd ~/adk_to_agent_engine/transcript_summarization_agent
python3 query_agent_engine.py

## View and delete deployed Agents
1. When your agent has completed its deployment, return to a browser tab showing the Cloud Console and navigate to Agent Engine by searching for it and selecting it at the top of the Console.
2. In the Region dropdown, make sure the Region us-central1 is selected.
3. You will see your deployed agent's display name. Select Service Configuration at the top of the Agent Engine console, and then the Deployment details tab.
4. From the Agent Engine Deployment details panel, copy the Resource name field, which will have a format like: projects/qwiklabs-gcp-02-76ce2eed15a5/locations/us-central1/reasoningEngines/1467742469964693504.
5. In the Cloud Shell Terminal, paste the following command and run:
cd ~/adk_to_agent_engine
python3 util/agent_engine_utils.py delete <REPLACE_WITH_COPIED_RESOURCE_NAME>
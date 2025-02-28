# Import necessary packages
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM

# Model and project settings
model_id = 'mistralai/mixtral-8x7b-instruct-v01' # Directly specifying the model

# Set necessary parameters
parameters = {
    GenParams.MAX_NEW_TOKENS: 256,  # Specifying the max tokens you want to generate
    GenParams.TEMPERATURE: 0.5, # This randomness or creativity of the model's responses
}

project_id = "skills-network"

# Wrap up the model into WatsonxLLM inference
watsonx_llm = WatsonxLLM(
    model_id=model_id,
    url="https://us-south.ml.cloud.ibm.com",
    project_id=project_id,
    params=parameters,
)

# Get the query from the user input
query = input("Please enter your query: ")

# Print the generated response
print(watsonx_llm.invoke(query))
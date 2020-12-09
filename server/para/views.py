from django.shortcuts import render
from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import HttpResponse
import os
from model_setup import unilm_paraphrase_generator
# Create your views here.

config_path = "config.json"

generator = unilm_paraphrase_generator(config_path)

@api_view(["POST"])
def para_gen(request):

    if request.method == "POST":
        text = request.data["text"].strip()
        output = generator.decode(text)
        print(text, output)
        return Response(
            {"res": output, status=status.HTTP_200_OK}
        )





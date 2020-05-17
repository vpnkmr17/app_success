from django.forms import ModelForm,Textarea,Select
from .models import AppSuccess

class AppSuccessForm(ModelForm):
    class Meta:
        model=AppSuccess
        fields="__all__"

        widgets = {
            'Category':Select(attrs={'style':"width:200px ;height:30px;font-size:14pt;"}),
        }

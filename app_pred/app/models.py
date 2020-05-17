from django.db import models

# Create your models here.
class AppSuccess(models.Model):
    CATEGORY_CHOICES=(
    ("ART_AND_DESIGN", "ART_AND_DESIGN"), 
    ("AUTO_AND_VEHICLES", "AUTO_AND_VEHICLES"), 
    ("BEAUTY", "BEAUTY"),
    ("COMICS", "COMICS"), 
    ("COMMUNICATION", "COMMUNICATION"), 
    ("EDUCATION", "EDUCATION"), 
    ("ENTERTAINMENT", "ENTERTAINMENT"), 
    ("FOOD_AND_DRINK", "FOOD_AND_DRINK"), 
    ("HOUSE_AND_HOME", "HOUSE_AND_HOME"), 
    ("LIBRARIES", "LIBRARIES"), 
    )

    CONTENT_CHOICES=(
    ("Everyone", "Everyone"), 
    ("Teen", "Teen"), 
    ("Everyone 10+", "Everyone 10+"), 
    ("Mature 17+", "Mature 17+"), 
    ("Adults only 18+", "Adults only 18+"), 
    ("Unrated", "Unrated"), 
    )

    App=models.CharField(max_length=150)
    Category=models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    Rating=models.FloatField()
    Reviews=models.CharField(max_length=250)
    Size=models.CharField(max_length=250)
    Type=models.CharField(max_length=250)
    Price=models.CharField(max_length=250)
    ContentRating=models.CharField(max_length=20, choices=CONTENT_CHOICES)
    Genres=models.CharField(max_length=100)
    LastUpdated=models.CharField(max_length=100)
    ContentVer=models.CharField(max_length=100)
    AndroidVer=models.CharField(max_length=100)

    def __str__(self):
        return self.App

    def to_dict(self):
        return {
        'App':self.App,
        'Category':self.Category,
        'Rating':self.Rating,
        'Reviews':self.Reviews,
        'Size':self.Size,
        'Type':self.Type,
        'Price':self.Price,
        'Content Rating':self.ContentRating,
        'Genres':self.Genres,
        'Last Updated':self.LastUpdated,
        'Current Ver':self.ContentVer,
        'Android Ver':self.AndroidVer
        }



    
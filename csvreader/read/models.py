from django.db import models
import os

class Dataset(models.Model):
    name = models.CharField(max_length=100, blank=True)
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    # ML Results save garna (Optional for now)
    target_col = models.CharField(max_length=50, blank=True, null=True)

    def save(self, *args, **kwargs):
        # File upload garda name xaina vane filename nai name banxa
        if not self.name and self.file:
            self.name = os.path.basename(self.file.name)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name
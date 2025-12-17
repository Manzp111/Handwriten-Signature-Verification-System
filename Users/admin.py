from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import User

@admin.register(User)
class CustomUserAdmin(BaseUserAdmin):
    # 1. Table Columns (List View)
    list_display = ('id', 'full_name', 'email', 'is_active', 'is_admin', 'created_at')
    list_filter = ('is_admin', 'is_active', 'is_superuser')
    
    # 2. Search and Ordering
    search_fields = ('email', 'full_name', 'uuid')
    ordering = ('-created_at',)
    
    # 3. Fieldsets (Detail View)
    # This replaces the default UserAdmin fields since we don't have 'username', etc.
    fieldsets = (
        ('Identity Profile', {
            'fields': ('uuid', 'full_name', 'email', 'password')
        }),
        ('Access Control', {
            'fields': ('is_active', 'is_admin', 'is_superuser')
        }),
        ('Metadata', {
            'fields': ('last_login', 'created_at')
        }),
    )

    # 4. Read-only fields
    readonly_fields = ('uuid', 'created_at', 'last_login')

    # Important: Since we don't use the standard 'username', 
    # we need to override these to keep the Admin from crashing
    filter_horizontal = ()
    list_per_page = 25
#!/usr/bin/env python3
"""
ArchNeuronX v4.0 Configuration Loader
Secure environment variable substitution for YAML files
"""

import os
import re
import sys
import yaml
from pathlib import Path

def substitute_env_vars(text):
    """Replace ${VAR} patterns with environment variables"""
    pattern = re.compile(r'\$\{([^}]+)\}')
    
    def replace_match(match):
        var_name = match.group(1)
        value = os.getenv(var_name)
        if value is None:
            print(f"⚠️  Environment variable {var_name} not found!")
            return f"${{{var_name}}}"  # Return original if not found
        return value
    
    return pattern.sub(replace_match, text)

def load_config(template_path, output_path):
    """Load template YAML and substitute environment variables"""
    try:
        # Read template file
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        # Substitute environment variables
        processed_content = substitute_env_vars(template_content)
        
        # Parse YAML to validate
        config = yaml.safe_load(processed_content)
        
        # Write processed configuration
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"✅ Configuration loaded: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False

def main():
    """Main function"""
    if len(sys.argv) != 3:
        print("Usage: python load_config.py <template_file> <output_file>")
        sys.exit(1)
    
    template_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Check if template file exists
    if not os.path.exists(template_file):
        print(f"❌ Template file not found: {template_file}")
        sys.exit(1)
    
    # Load environment variables from .env file
    env_file = Path('.env')
    if env_file.exists():
        print("📁 Loading environment variables from .env file...")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("✅ Environment variables loaded")
    else:
        print("⚠️  .env file not found. Using system environment variables.")
    
    # Load configuration
    success = load_config(template_file, output_file)
    
    if success:
        print("🎉 Configuration loaded successfully!")
        
        # Verify critical variables
        critical_vars = ['BINANCE_API_KEY', 'BINANCE_API_SECRET']
        missing_vars = []
        
        for var in critical_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            print(f"⚠️  Missing critical environment variables: {', '.join(missing_vars)}")
            print("📝 Please update your .env file with these variables.")
        else:
            print("✅ All critical environment variables are set!")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

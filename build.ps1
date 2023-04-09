Write-Host -ForegroundColor Blue "Making virtual env"
Start-Process -Wait -NoNewWindow -FilePath "python" -ArgumentList "-m","venv","twinkle","--copies"
Start-Process -Wait -NoNewWindow -FilePath "twinkle/Scripts/pip"  -ArgumentList "install","-r","requirements.txt"
Start-Process -Wait -NoNewWindow -FilePath "twinkle/Scripts/pip"  -ArgumentList "install","--force-reinstall","PyOpenGL-3.1.6-cp310-cp310-win_amd64.whl"

Write-Host -ForegroundColor Blue "Installing demo"
Copy-Item -Path "demo.py","transforms.py","fbo.py" -Destination "twinkle"
New-Item -ItemType "Directory" -Path "twinkle\assembly" -Force
Copy-Item -Path "assembly\*py" -Destination "twinkle\assembly"
New-Item -ItemType "Directory" -Path "twinkle\geometry" -Force
Copy-Item -Path "geometry\*py" -Destination "twinkle\geometry"

Write-Host -ForegroundColor Blue "Building executable"
Start-Process -Wait -NoNewWindow -FilePath "twinkle\Scripts\pyinstaller" -ArgumentList "--onefile","demo.py" -WorkingDirectory "twinkle"

Write-Host -ForegroundColor Blue "Packaging demo"
Compress-Archive -Update -LiteralPath "fm.mp4","text.gltf","twinkle\dist\demo.exe" -DestinationPath "twinkle-demo.zip" -CompressionLevel "Optimal"

Write-Host -ForegroundColor Green "Cleaning up"
Remove-Item -Force -Recurse -Path "twinkle"

Write-Host -ForegroundColor Green "Done"

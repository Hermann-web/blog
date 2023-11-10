cd web
cd ..
deactivate
rm -rf venv/
conda activate web-dev
python -m venv venv
source venv/bin/activate
git checkout source && git add . && git commit -m "pre-prod" && git push
cd web

pip install -r requirements-full.txt
git branch -D master
git checkout source
git checkout -b master
mkdocs build
mv site ../tmp-dir
cd ..
rm -r web
mv tmp-dir web
git add .
git commit -m "deploy"
git push --set-upstream origin master --force

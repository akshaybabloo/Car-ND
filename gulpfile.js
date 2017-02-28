var gulp = require('gulp');
var shell = require('gulp-shell');


gulp.task('make_html', shell.task('make html', {cwd: './Docs'}));

gulp.task('default', ['make_html']);

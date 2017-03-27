var gulp = require('gulp');
var exec = require('gulp-exec');


gulp.task('make_html', function (cb) {
    process.chdir("./docs");
    var reportOptions = {
        err: true, // default = true, false means don't write err
        stderr: true, // default = true, false means don't write stderr
        stdout: true // default = true, false means don't write stdout
    };
    return gulp.src(".")
        .pipe(exec('make clean'))
        .pipe(exec('make html'))
        .pipe(exec.reporter(reportOptions))
});

gulp.task('default', ['make_html']);

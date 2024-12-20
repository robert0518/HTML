#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <mpfr.h>
#define YEAR_NOW 2024
#define MONTH_NOW 1
#define DAY_NOW 1

double alpha = 5e-4;

int team_hash(char* str) {
    int sum = 0;
    for (int i = 0; i < 3; i++) {
        sum += str[i] * (i + 1);
    }
    sum %= 100;
    return sum;
}

int parse_bool(char* str) {
    if (!strcmp(str, "True")) return 1;
    else return 0;
}

int date_dif(int year, int month, int day) {
    struct tm date1 = { 0 }, date2 = { 0 };
    time_t time1, time2;

    date1.tm_year = year - 1900;
    date1.tm_mon = month - 1;
    date1.tm_mday = day;

    date2.tm_year = YEAR_NOW - 1900;
    date2.tm_mon = MONTH_NOW - 1;
    date2.tm_mday = DAY_NOW;

    time1 = mktime(&date1);
    time2 = mktime(&date2);

    double difference = difftime(time2, time1) / (60 * 60 * 24);
    return (int)difference;
}

mpfr_t* decay(int day_dif) {
    mpfr_t* ret = (mpfr_t*)malloc(sizeof(mpfr_t));
    mpfr_t tmp;
    mpfr_init2(*ret, 128);
    mpfr_init2(tmp, 128);
    mpfr_set_d(tmp, -1.0 * alpha * day_dif, MPFR_RNDN);
    mpfr_exp(*ret, tmp, MPFR_RNDN);
    mpfr_clear(tmp);
    return ret;
}

int main() {
    FILE* train = fopen("../train_data.csv", "r");
    FILE* out = fopen("train_data_libsvm_alpha_0.0005_mean", "w");
    // FILE* list = fopen("libsvm_list", "w");

    char buffer[7000];

    // parse list
    fgets(buffer, sizeof(buffer), train);
    // char* name = buffer;
    // char* comma = strchr(name, ',');
    // int num = 1;
    // while (comma) {
    //     *comma = 0;
    //     fprintf(list, "%3.d %s\n", num++, name);
    //     name = comma + 1;
    //     comma = strchr(name, ',');
    // }
    // fclose(list);

    // parse data
    char* name, * comma;
    while (fgets(buffer, sizeof(buffer), train)) {
        name = buffer;
        comma = strchr(name, ',');
        int num = 1, temp;
        mpfr_t* decay_coef;
        while (comma) {
            *comma = 0;

            if (num == 1) {
            } else if (num == 2) {
                fprintf(out, "home:%-2d ", team_hash(name));
            } else if (num == 3) {
                fprintf(out, "away:%-2d ", team_hash(name));
            } else if (num == 4) {
                char* year = name;
                char* month = strchr(year, '-');
                *month = 0;
                month++;
                char* day = strchr(month, '-');
                *day = 0;
                day++;
                int day_dif = date_dif(atoi(year), atoi(month), atoi(day));
                decay_coef = decay(day_dif);
                fprintf(out, "decay:");
                mpfr_fprintf(out, "%.20Rf", decay_coef);
                fprintf(out, "|");
            } else if (num == 5) {
                temp = parse_bool(name);
            } else if (num == 6) {
                fprintf(out, "%d ", parse_bool(name));
                if (temp)
                    mpfr_fprintf(out, "1:%.20Rf ", decay_coef);
            } else if (num == 7 || num == 8) {
            } else {
                if (name != comma) {
                    if (strcmp(name, "0.0") && (num < 42 || (num >= 42 && num % 3 == 0))) {
                        fprintf(out, "%d:", num - 7);
                        mpfr_t tmp;
                        mpfr_init2(tmp, 128);
                        mpfr_set_str(tmp, name, 10, MPFR_RNDN);
                        mpfr_mul(tmp, tmp, *decay_coef, MPFR_RNDN);
                        mpfr_fprintf(out, "%.20Rf ", tmp);
                        mpfr_clear(tmp);
                    }
                }
            }
            name = comma + 1;
            comma = strchr(name, ',');
            num++;
        }
        comma = strchr(name, '\n');
        *comma = 0;
        if (name != comma && (num < 42 || (num >= 42 && num % 3 == 0))) {
            fprintf(out, "%d:", num - 7);
            mpfr_t tmp;
            mpfr_init2(tmp, 128);
            mpfr_set_str(tmp, name, 10, MPFR_RNDN);
            mpfr_mul(tmp, tmp, *decay_coef, MPFR_RNDN);
            mpfr_fprintf(out, "%.20Rf ", tmp);
            mpfr_clear(tmp);
        }
        fprintf(out, "\n");
        mpfr_clear(*decay_coef);
    }

    fclose(train);
    fclose(out);


    return 0;
}
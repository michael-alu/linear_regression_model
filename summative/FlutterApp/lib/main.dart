import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';

const String apiBaseUrl = 'https://linear-regression-api-3o1x.onrender.com';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.cyan),
      ),
      home: const MyHomePage(title: 'AI Job Market Salary Predictor'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  static const List<String> jobTitleOptions = [
    "AI Research Scientist",
    "AI Software Engineer",
    "AI Specialist",
    "NLP Engineer",
    "AI Consultant",
    "AI Architect",
    "Principal Data Scientist",
    "Data Analyst",
    "Autonomous Systems Engineer",
    "AI Product Manager",
    "Machine Learning Engineer",
    "Data Engineer",
    "Research Scientist",
    "ML Ops Engineer",
    "Robotics Engineer",
    "Head of AI",
    "Deep Learning Engineer",
    "Data Scientist",
    "Machine Learning Researcher",
    "Computer Vision Engineer",
  ];

  static const List<String> employmentTypeOptions = ["CT", "FL", "PT", "FT"];

  static const List<String> industryOptions = [
    "Automotive",
    "Media",
    "Education",
    "Consulting",
    "Healthcare",
    "Gaming",
    "Government",
    "Telecommunications",
    "Manufacturing",
    "Energy",
    "Technology",
    "Real Estate",
    "Finance",
    "Transportation",
    "Retail",
  ];

  static const List<String> countryOptions = [
    "China",
    "Canada",
    "Switzerland",
    "India",
    "France",
    "Germany",
    "United Kingdom",
    "Singapore",
    "Austria",
    "Sweden",
    "South Korea",
    "Norway",
    "Netherlands",
    "United States",
    "Israel",
    "Australia",
    "Ireland",
    "Denmark",
    "Finland",
    "Japan",
  ];

  static const List<String> experienceLevelOptions = ["EN", "MI", "SE", "EX"];

  static const List<String> companySizeOptions = ["S", "M", "L"];

  static const List<String> educationRequiredOptions = [
    "Associate",
    "Bachelor",
    "Master",
    "PhD",
  ];

  static const List<String> requiredSkillOptions = [
    "Python",
    "SQL",
    "TensorFlow",
    "Kubernetes",
    "Scala",
    "PyTorch",
    "Linux",
    "Git",
    "Java",
    "GCP",
    "Hadoop",
    "Tableau",
    "R",
    "Computer Vision",
    "Data Visualization",
    "Deep Learning",
    "MLOps",
    "Spark",
    "NLP",
    "Azure",
    "AWS",
    "Mathematics",
    "Docker",
    "Statistics",
  ];

  String? _jobTitle;
  String? _employmentType;
  String? _industry;
  String? _companyLocation;
  String? _employeeResidence;
  String? _experienceLevel;
  String? _companySize;
  String? _educationRequired;
  int? _remoteRatio;
  final Set<String> _selectedSkills = <String>{};

  final _yearsExperienceController = TextEditingController();
  final _jobDescriptionLengthController = TextEditingController();
  final _benefitsScoreController = TextEditingController();

  bool _isLoading = false;
  String? _errorMessage;

  @override
  void dispose() {
    _yearsExperienceController.dispose();
    _jobDescriptionLengthController.dispose();
    _benefitsScoreController.dispose();
    super.dispose();
  }

  Future<void> _predict() async {
    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    final jobTitle = _jobTitle;
    final employmentType = _employmentType;
    final industry = _industry;
    final companyLocation = _companyLocation;
    final employeeResidence = _employeeResidence;
    final experienceLevel = _experienceLevel;
    final companySize = _companySize;
    final educationRequired = _educationRequired;

    final numericYearsExperience = int.tryParse(
      _yearsExperienceController.text.trim(),
    );
    final numericJobDescriptionLength = int.tryParse(
      _jobDescriptionLengthController.text.trim(),
    );
    final numericBenefitsScore = double.tryParse(
      _benefitsScoreController.text.trim(),
    );

    if (jobTitle == null ||
        employmentType == null ||
        industry == null ||
        companyLocation == null ||
        employeeResidence == null ||
        experienceLevel == null ||
        companySize == null ||
        educationRequired == null) {
      setState(() {
        _isLoading = false;
        _errorMessage = 'Please select all dropdown values.';
      });
      return;
    }

    final numericErrors = <String>[];
    if (numericYearsExperience == null) {
      numericErrors.add('Years of experience must be an integer.');
    }
    if (numericJobDescriptionLength == null) {
      numericErrors.add('Job description length must be an integer.');
    }
    if (numericBenefitsScore == null) {
      numericErrors.add('Benefits score must be a number.');
    }

    if (numericErrors.isNotEmpty) {
      setState(() {
        _isLoading = false;
        _errorMessage = numericErrors.join('\n');
      });
      return;
    }

    if (_remoteRatio == null) {
      setState(() {
        _isLoading = false;
        _errorMessage = 'Please select remote ratio (0, 50, or 100).';
      });
      return;
    }
    if (numericYearsExperience! < 0 || numericYearsExperience > 50) {
      setState(() {
        _isLoading = false;
        _errorMessage = 'Years of experience must be between 0 and 50.';
      });
      return;
    }
    if (numericJobDescriptionLength! < 0 ||
        numericJobDescriptionLength > 20000) {
      setState(() {
        _isLoading = false;
        _errorMessage = 'Job description length must be between 0 and 20000.';
      });
      return;
    }
    if (numericBenefitsScore! < 0.0 || numericBenefitsScore > 10.0) {
      setState(() {
        _isLoading = false;
        _errorMessage = 'Benefits score must be between 0.0 and 10.0.';
      });
      return;
    }

    final payload = <String, Object?>{
      'job_title': jobTitle,
      'employment_type': employmentType,
      'industry': industry,
      'company_location': companyLocation,
      'employee_residence': employeeResidence,
      'experience_level': experienceLevel,
      'company_size': companySize,
      'education_required': educationRequired,
      'remote_ratio': _remoteRatio,
      'years_experience': numericYearsExperience,
      'job_description_length': numericJobDescriptionLength,
      'benefits_score': numericBenefitsScore,
      'required_skills': _selectedSkills.toList(),
    };

    try {
      final uri = Uri.parse('$apiBaseUrl/predict');
      final client = HttpClient();
      final request = await client.postUrl(uri);
      request.headers.contentType = ContentType.json;
      request.write(jsonEncode(payload));

      final response = await request.close();
      final responseBody = await response.transform(utf8.decoder).join();

      if (response.statusCode != 200) {
        throw Exception(
          'API error (${response.statusCode}): ${responseBody.isEmpty ? "Unknown error" : responseBody}',
        );
      }

      final decoded = jsonDecode(responseBody) as Map<String, dynamic>;
      final predicted = decoded['predicted_salary_usd'];
      if (predicted is num) {
        setState(() {
          _errorMessage = null;
          _isLoading = false;
        });
        await _showPredictionDialog(predicted.toDouble());
      } else {
        throw Exception('Unexpected API response: $decoded');
      }
    } catch (e) {
      setState(() {
        _errorMessage = e.toString();
      });
    } finally {
      if (_isLoading) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  Widget _field({
    required String label,
    required TextEditingController controller,
    TextInputType keyboardType = TextInputType.text,
  }) {
    return TextField(
      controller: controller,
      keyboardType: keyboardType,
      decoration: InputDecoration(
        labelText: label,
        border: const OutlineInputBorder(),
      ),
    );
  }

  Widget _dropdownField({
    required String label,
    required List<String> options,
    required String? value,
    required ValueChanged<String?> onChanged,
  }) {
    return DropdownButtonFormField<String>(
      decoration: InputDecoration(
        labelText: label,
        border: const OutlineInputBorder(),
      ),
      hint: Text('Select $label'),
      value: value,
      items: options
          .map((s) => DropdownMenuItem<String>(value: s, child: Text(s)))
          .toList(),
      onChanged: onChanged,
    );
  }

  Widget _skillsChipSelector() {
    return InputDecorator(
      decoration: const InputDecoration(
        labelText: 'Required Skills (select multiple)',
        border: OutlineInputBorder(),
      ),
      child: Wrap(
        spacing: 8,
        runSpacing: 8,
        children: requiredSkillOptions.map((skill) {
          final isSelected = _selectedSkills.contains(skill);
          return FilterChip(
            label: Text(skill),
            selected: isSelected,
            onSelected: (selected) {
              setState(() {
                if (selected) {
                  _selectedSkills.add(skill);
                } else {
                  _selectedSkills.remove(skill);
                }
              });
            },
          );
        }).toList(),
      ),
    );
  }

  Future<void> _showPredictionDialog(double predictedSalaryUsd) async {
    await showDialog<void>(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Prediction Result'),
          content: Text(
            'Predicted Salary:\n \$${_formatCurrency(predictedSalaryUsd)}',
            style: Theme.of(context).textTheme.titleMedium,
            textAlign: TextAlign.center,
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Close'),
            ),
          ],
        );
      },
    );
  }

  String _formatCurrency(double value) {
    final parts = value.toStringAsFixed(2).split('.');
    final integer = parts.first;
    final decimal = parts.last;
    final withCommas = integer.replaceAllMapped(
      RegExp(r'\B(?=(\d{3})+(?!\d))'),
      (_) => ',',
    );
    return '$withCommas.$decimal';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text('Salary Prediction'),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              _dropdownField(
                label: 'Job Title',
                options: jobTitleOptions,
                value: _jobTitle,
                onChanged: (v) => setState(() => _jobTitle = v),
              ),
              const SizedBox(height: 12),
              _dropdownField(
                label: 'Employment Type (CT/FL/PT/FT)',
                options: employmentTypeOptions,
                value: _employmentType,
                onChanged: (v) => setState(() => _employmentType = v),
              ),
              const SizedBox(height: 12),
              _dropdownField(
                label: 'Industry',
                options: industryOptions,
                value: _industry,
                onChanged: (v) => setState(() => _industry = v),
              ),
              const SizedBox(height: 12),
              _dropdownField(
                label: 'Company Location',
                options: countryOptions,
                value: _companyLocation,
                onChanged: (v) => setState(() => _companyLocation = v),
              ),
              const SizedBox(height: 12),
              _dropdownField(
                label: 'Employee Residence',
                options: countryOptions,
                value: _employeeResidence,
                onChanged: (v) => setState(() => _employeeResidence = v),
              ),
              const SizedBox(height: 12),
              _dropdownField(
                label: 'Experience Level (EN/MI/SE/EX)',
                options: experienceLevelOptions,
                value: _experienceLevel,
                onChanged: (v) => setState(() => _experienceLevel = v),
              ),
              const SizedBox(height: 12),
              _dropdownField(
                label: 'Company Size (S/M/L)',
                options: companySizeOptions,
                value: _companySize,
                onChanged: (v) => setState(() => _companySize = v),
              ),
              const SizedBox(height: 12),
              _dropdownField(
                label: 'Education Required',
                options: educationRequiredOptions,
                value: _educationRequired,
                onChanged: (v) => setState(() => _educationRequired = v),
              ),
              const SizedBox(height: 12),
              _dropdownField(
                label: 'Remote Ratio',
                options: const ['0', '50', '100'],
                value: _remoteRatio?.toString(),
                onChanged: (v) =>
                    setState(() => _remoteRatio = int.tryParse(v ?? '')),
              ),
              const SizedBox(height: 12),
              _skillsChipSelector(),
              const SizedBox(height: 12),
              _field(
                label: 'Years of Experience (0-50)',
                controller: _yearsExperienceController,
                keyboardType: TextInputType.number,
              ),
              const SizedBox(height: 12),
              _field(
                label: 'Job Description Length (0-20000)',
                controller: _jobDescriptionLengthController,
                keyboardType: TextInputType.number,
              ),
              const SizedBox(height: 12),
              _field(
                label: 'Benefits Score (0.0-10.0)',
                controller: _benefitsScoreController,
                keyboardType: const TextInputType.numberWithOptions(
                  decimal: true,
                ),
              ),
              const SizedBox(height: 16),
              SizedBox(
                height: 48,
                child: ElevatedButton(
                  onPressed: _isLoading ? null : _predict,
                  child: Text(_isLoading ? 'Predicting...' : 'Predict'),
                ),
              ),
              const SizedBox(height: 16),
              if (_errorMessage != null) ...[
                const SizedBox(height: 8),
                Text(
                  'Error: $_errorMessage',
                  style: const TextStyle(color: Colors.red),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

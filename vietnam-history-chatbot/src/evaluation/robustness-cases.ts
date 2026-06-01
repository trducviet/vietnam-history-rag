/**
 * Robustness Test Cases — 60 cases for generalization testing
 * beyond the core 20 answer-quality benchmark.
 *
 * Patch 7I: New file.
 * Patch 7I-A: Stricter schema — two-sided comparison, clarification_expected,
 *   answer style, treaty-specific guards.
 */

// ─── Types ───────────────────────────────────────────────────

export type RobustnessCategory =
  | 'date_lookup' | 'actor_lookup' | 'location_lookup'
  | 'organization_lookup' | 'treaty_lookup' | 'clause_lookup'
  | 'timeline' | 'comparison' | 'disambiguation'
  | 'misconception_check' | 'explanation' | 'multi_hop'
  | 'out_of_scope' | 'paraphrase' | 'adversarial';

export type CitationBehavior =
  | 'at_least_one' | 'primary_only' | 'allow_multiple'
  | 'no_citation_for_oos' | 'no_citation_for_ambiguous';

export type FailureMode =
  | 'routing_error' | 'retrieval_error' | 'evidence_role_error'
  | 'citation_error' | 'answer_wording_error' | 'scope_error'
  | 'weak_grounding' | 'dataset_gap' | 'acceptable'
  | 'false_positive' | 'one_sided_comparison' | 'missing_clarification';

export type AnswerStyle =
  | 'direct_lookup' | 'comparison_two_sided' | 'misconception_correction'
  | 'scope_refusal' | 'clarification_needed' | 'timeline' | 'explanation';

export interface RobustnessCase {
  id: string;
  category: RobustnessCategory;
  difficulty: 'easy' | 'medium' | 'hard';
  query: string;
  expected_behavior: string;
  should_refuse?: boolean;
  expected_intent?: string;
  required_terms_any?: string[];
  required_terms_all?: string[];
  forbidden_terms?: string[];
  expected_citation_behavior?: CitationBehavior;
  notes?: string;

  // Patch 7I-A stricter fields
  expected_answer_style?: AnswerStyle;
  required_terms_both_sides?: { side_a: string[]; side_b: string[] };
  must_include_citations_from_both_sides?: boolean;
  must_not_cite_if_oos?: boolean;
  clarification_expected?: boolean;
  required_citation_terms_any?: string[];
  forbidden_citation_terms?: string[];
}

// ─── Cases ───────────────────────────────────────────────────

export const ROBUSTNESS_CASES: RobustnessCase[] = [
  // ══════ DATE LOOKUP (6) ══════
  { id: 'rb_date_001', category: 'date_lookup', difficulty: 'easy',
    query: 'Chiến dịch Hồ Chí Minh bắt đầu khi nào?',
    expected_behavior: 'Return year 1975',
    required_terms_any: ['1975'], expected_citation_behavior: 'at_least_one' },
  { id: 'rb_date_002', category: 'date_lookup', difficulty: 'easy',
    query: 'Pháp nổ súng xâm lược Việt Nam năm nào?',
    expected_behavior: 'Return 1858 or 1-9-1858',
    required_terms_any: ['1858'], expected_citation_behavior: 'at_least_one' },
  { id: 'rb_date_003', category: 'date_lookup', difficulty: 'easy',
    query: 'Cách mạng tháng Tám diễn ra năm nào?',
    expected_behavior: 'Return 1945',
    required_terms_any: ['1945'], expected_citation_behavior: 'at_least_one' },
  { id: 'rb_date_004', category: 'date_lookup', difficulty: 'medium',
    query: 'Chiến thắng Điện Biên Phủ diễn ra vào ngày tháng năm nào?',
    expected_behavior: 'Return 7-5-1954 or May 1954',
    required_terms_any: ['1954', '7-5-1954', '7/5/1954', 'tháng 5'], expected_citation_behavior: 'at_least_one' },
  { id: 'rb_date_005', category: 'date_lookup', difficulty: 'medium',
    query: 'Việt Nam thống nhất đất nước chính thức vào năm nào?',
    expected_behavior: 'Return 1976 or 1975',
    required_terms_any: ['1975', '1976'], expected_citation_behavior: 'at_least_one' },
  { id: 'rb_date_006', category: 'date_lookup', difficulty: 'medium',
    query: 'Đảng Cộng sản Việt Nam được thành lập năm nào?',
    expected_behavior: 'Return 1930',
    required_terms_any: ['1930'], expected_citation_behavior: 'at_least_one' },

  // ══════ ACTOR LOOKUP (5) ══════
  { id: 'rb_actor_001', category: 'actor_lookup', difficulty: 'easy',
    query: 'Ai lãnh đạo chiến dịch Điện Biên Phủ phía Việt Nam?',
    expected_behavior: 'Mention Võ Nguyên Giáp',
    required_terms_any: ['Võ Nguyên Giáp', 'Giáp'], expected_citation_behavior: 'at_least_one' },
  { id: 'rb_actor_002', category: 'actor_lookup', difficulty: 'easy',
    query: 'Ai thành lập Đảng Cộng sản Việt Nam?',
    expected_behavior: 'Mention Nguyễn Ái Quốc / Hồ Chí Minh',
    required_terms_any: ['Nguyễn Ái Quốc', 'Hồ Chí Minh'], expected_citation_behavior: 'at_least_one' },
  { id: 'rb_actor_003', category: 'actor_lookup', difficulty: 'medium',
    query: 'Ai ký Hiệp định Genève phía Việt Nam?',
    expected_behavior: 'Mention relevant Vietnamese delegation',
    expected_citation_behavior: 'at_least_one' },
  { id: 'rb_actor_004', category: 'actor_lookup', difficulty: 'medium',
    query: 'Tổng thống nào của Việt Nam Cộng hòa bị lật đổ năm 1963?',
    expected_behavior: 'Mention Ngô Đình Diệm',
    required_terms_any: ['Ngô Đình Diệm', 'Diệm'], expected_citation_behavior: 'at_least_one' },
  { id: 'rb_actor_005', category: 'actor_lookup', difficulty: 'hard',
    query: 'Ai là người đứng đầu phái đoàn Pháp tại Điện Biên Phủ?',
    expected_behavior: 'Mention de Castries or Navarre',
    required_terms_any: ['Castries', 'Navarre', 'Pháp'], expected_citation_behavior: 'at_least_one' },

  // ══════ LOCATION LOOKUP (5) ══════
  { id: 'rb_loc_001', category: 'location_lookup', difficulty: 'easy',
    query: 'Tuyên ngôn Độc lập được đọc tại đâu?',
    expected_behavior: 'Mention Ba Đình or Hà Nội',
    required_terms_any: ['Ba Đình', 'Hà Nội'], expected_citation_behavior: 'at_least_one' },
  { id: 'rb_loc_002', category: 'location_lookup', difficulty: 'easy',
    query: 'Pháp tấn công cảng nào đầu tiên khi xâm lược Việt Nam?',
    expected_behavior: 'Mention Đà Nẵng or Tourane',
    required_terms_any: ['Đà Nẵng', 'Tourane', 'cảng'], expected_citation_behavior: 'at_least_one' },
  { id: 'rb_loc_003', category: 'location_lookup', difficulty: 'medium',
    query: 'Hội nghị thành lập Đảng Cộng sản Việt Nam diễn ra ở đâu?',
    expected_behavior: 'Mention Hong Kong or Cửu Long',
    required_terms_any: ['Hương Cảng', 'Hong Kong', 'Cửu Long'], expected_citation_behavior: 'at_least_one' },
  { id: 'rb_loc_004', category: 'location_lookup', difficulty: 'medium',
    query: 'Hiệp định Genève được ký tại đâu?',
    expected_behavior: 'Mention Genève / Geneva',
    required_terms_any: ['Genève', 'Geneva', 'Giơnevơ'], expected_citation_behavior: 'at_least_one' },
  { id: 'rb_loc_005', category: 'location_lookup', difficulty: 'hard',
    query: 'Chiến dịch Biên giới 1950 diễn ra tại khu vực nào?',
    expected_behavior: 'Mention biên giới Việt-Trung or Cao Bằng/Lạng Sơn',
    required_terms_any: ['Cao Bằng', 'Lạng Sơn', 'biên giới'], expected_citation_behavior: 'at_least_one' },

  // ══════ ORGANIZATION LOOKUP (5) ══════
  { id: 'rb_org_001', category: 'organization_lookup', difficulty: 'easy',
    query: 'Mặt trận nào ra đời năm 1941?',
    expected_behavior: 'Mention Việt Minh',
    required_terms_any: ['Việt Minh', 'Việt Nam Độc lập Đồng minh Hội'], expected_citation_behavior: 'at_least_one' },
  { id: 'rb_org_002', category: 'organization_lookup', difficulty: 'easy',
    query: 'Tổ chức quốc tế nào Việt Nam gia nhập năm 1995?',
    expected_behavior: 'Mention ASEAN',
    required_terms_any: ['ASEAN'], expected_citation_behavior: 'at_least_one' },
  { id: 'rb_org_003', category: 'organization_lookup', difficulty: 'medium',
    query: 'Ai thành lập Mặt trận Tổ quốc Việt Nam?',
    expected_behavior: 'Provide information about Mặt trận Tổ quốc',
    expected_citation_behavior: 'at_least_one' },
  { id: 'rb_org_004', category: 'organization_lookup', difficulty: 'medium',
    query: 'Chính phủ lâm thời nước Việt Nam Dân chủ Cộng hòa được thành lập khi nào?',
    expected_behavior: 'Return 1945',
    required_terms_any: ['1945'], expected_citation_behavior: 'at_least_one' },
  { id: 'rb_org_005', category: 'organization_lookup', difficulty: 'hard',
    query: 'Liên minh quân sự nào do Pháp lập ra trước Điện Biên Phủ?',
    expected_behavior: 'Provide relevant context about French military',
    expected_citation_behavior: 'at_least_one' },

  // ══════ TREATY / CLAUSE (8) ══════
  { id: 'rb_treaty_001', category: 'treaty_lookup', difficulty: 'easy',
    query: 'Hiệp định Genève quy định ranh giới tạm thời ở đâu?',
    expected_behavior: 'Mention vĩ tuyến 17',
    required_terms_any: ['vĩ tuyến 17', 'vĩ tuyến'], expected_citation_behavior: 'at_least_one' },
  { id: 'rb_treaty_002', category: 'treaty_lookup', difficulty: 'easy',
    query: 'Hiệp định Paris quy định điều gì cho quân Mỹ?',
    expected_behavior: 'Mention rút quân / ngừng bắn — must be about Paris, not Genève',
    expected_answer_style: 'direct_lookup',
    required_terms_any: ['rút quân', 'ngừng bắn', 'chấm dứt'], expected_citation_behavior: 'at_least_one' },
  { id: 'rb_treaty_003', category: 'clause_lookup', difficulty: 'medium',
    query: 'Tổng tuyển cử thống nhất theo Hiệp định Genève dự kiến năm nào?',
    expected_behavior: 'Return 1956',
    expected_answer_style: 'direct_lookup',
    required_terms_all: ['tổng tuyển cử'], required_terms_any: ['1956'], expected_citation_behavior: 'at_least_one' },
  { id: 'rb_treaty_004', category: 'treaty_lookup', difficulty: 'medium',
    query: 'Hiệp ước Nhâm Tuất 1862 quy định gì?',
    expected_behavior: 'Mention cắt/nhượng đất or Nam Kỳ',
    expected_answer_style: 'direct_lookup',
    required_terms_any: ['1862', 'Nhâm Tuất'], expected_citation_behavior: 'at_least_one' },
  { id: 'rb_treaty_005', category: 'treaty_lookup', difficulty: 'medium',
    query: 'Hiệp định Paris có quy định tổng tuyển cử không?',
    expected_behavior: 'Clarify Paris provisions — should not conflate with Genève tổng tuyển cử',
    expected_answer_style: 'misconception_correction',
    required_terms_any: ['Paris', 'không', 'ngừng bắn', 'rút quân'],
    expected_citation_behavior: 'at_least_one' },
  { id: 'rb_treaty_006', category: 'clause_lookup', difficulty: 'hard',
    query: 'Điều khoản nào của Hiệp định Genève nói về tập kết chuyển quân?',
    expected_behavior: 'Mention tập kết / chuyển quân / 300 ngày',
    required_terms_any: ['tập kết', 'chuyển quân'], expected_citation_behavior: 'at_least_one' },
  { id: 'rb_treaty_007', category: 'treaty_lookup', difficulty: 'hard',
    query: 'Hòa ước Patenôtre 1884 có ý nghĩa gì?',
    expected_behavior: 'Mention bảo hộ / thực dân',
    required_terms_any: ['1884', 'Patenôtre', 'bảo hộ'], expected_citation_behavior: 'at_least_one' },
  { id: 'rb_treaty_008', category: 'treaty_lookup', difficulty: 'hard',
    query: 'Hiệp định sơ bộ 6-3-1946 giữa ai với ai?',
    expected_behavior: 'Mention Việt Nam and Pháp',
    required_terms_any: ['1946', 'Pháp'], expected_citation_behavior: 'at_least_one' },

  // ══════ TIMELINE (5) ══════
  { id: 'rb_timeline_001', category: 'timeline', difficulty: 'medium',
    query: 'Tóm tắt các mốc chính của chiến dịch Điện Biên Phủ.',
    expected_behavior: 'List chronological milestones with dates',
    required_terms_any: ['Điện Biên Phủ', '1954'], expected_citation_behavior: 'allow_multiple' },
  { id: 'rb_timeline_002', category: 'timeline', difficulty: 'medium',
    query: 'Liệt kê các sự kiện chính từ 1945 đến 1954.',
    expected_behavior: 'Chronological list covering the First Indochina War period',
    required_terms_any: ['1945', '1954'], expected_citation_behavior: 'allow_multiple' },
  { id: 'rb_timeline_003', category: 'timeline', difficulty: 'hard',
    query: 'Nêu các mốc quan trọng của phong trào Đổi Mới.',
    expected_behavior: 'Mention 1986 and reform milestones',
    required_terms_any: ['Đổi Mới', '1986', 'đổi mới'], expected_citation_behavior: 'allow_multiple' },
  { id: 'rb_timeline_004', category: 'timeline', difficulty: 'hard',
    query: 'Các bước leo thang chiến tranh của Mỹ ở Việt Nam diễn ra thế nào?',
    expected_behavior: 'Chronological escalation events',
    expected_citation_behavior: 'allow_multiple' },
  { id: 'rb_timeline_005', category: 'timeline', difficulty: 'medium',
    query: 'Tóm tắt quá trình Pháp xâm lược Việt Nam.',
    expected_behavior: 'Mention key dates from 1858 onward',
    required_terms_any: ['1858', 'Pháp'], expected_citation_behavior: 'allow_multiple' },

  // ══════ COMPARISON (6) ══════
  { id: 'rb_compare_001', category: 'comparison', difficulty: 'medium',
    query: 'So sánh Hiệp định Genève và Hiệp định Paris.',
    expected_behavior: 'Compare both treaties — must mention both',
    expected_answer_style: 'comparison_two_sided',
    required_terms_all: ['Genève', 'Paris'],
    required_terms_both_sides: { side_a: ['Genève'], side_b: ['Paris'] },
    expected_citation_behavior: 'allow_multiple' },
  { id: 'rb_compare_002', category: 'comparison', difficulty: 'hard',
    query: 'Chiến dịch Biên giới 1950 khác gì với chiến dịch Điện Biên Phủ?',
    expected_behavior: 'Compare two campaigns — must cover both sides',
    expected_answer_style: 'comparison_two_sided',
    required_terms_both_sides: { side_a: ['Biên giới', '1950'], side_b: ['Điện Biên Phủ', '1954'] },
    expected_citation_behavior: 'allow_multiple' },
  { id: 'rb_compare_003', category: 'comparison', difficulty: 'medium',
    query: 'Tổng khởi nghĩa tháng Tám 1945 khác gì với chiến dịch Hồ Chí Minh 1975?',
    expected_behavior: 'Compare both events — must cover both sides',
    expected_answer_style: 'comparison_two_sided',
    required_terms_both_sides: { side_a: ['tháng Tám', '1945'], side_b: ['Hồ Chí Minh', '1975'] },
    expected_citation_behavior: 'allow_multiple' },
  { id: 'rb_compare_004', category: 'comparison', difficulty: 'hard',
    query: 'Hiến pháp 1946 khác gì so với Hiến pháp 1959?',
    expected_behavior: 'Compare constitutions — must mention both years',
    expected_answer_style: 'comparison_two_sided',
    required_terms_both_sides: { side_a: ['1946'], side_b: ['1959'] },
    forbidden_terms: ['1980'],
    expected_citation_behavior: 'allow_multiple' },
  { id: 'rb_compare_005', category: 'comparison', difficulty: 'medium',
    query: 'Gia nhập ASEAN và gia nhập APEC có giống nhau không?',
    expected_behavior: 'Distinguish ASEAN from APEC',
    expected_answer_style: 'comparison_two_sided',
    required_terms_all: ['ASEAN', 'APEC'],
    required_terms_both_sides: { side_a: ['ASEAN'], side_b: ['APEC'] },
    expected_citation_behavior: 'allow_multiple' },
  { id: 'rb_compare_006', category: 'comparison', difficulty: 'hard',
    query: 'Phong trào Xô viết Nghệ Tĩnh khác gì với phong trào Cần Vương?',
    expected_behavior: 'Compare two movements — must cover both sides',
    expected_answer_style: 'comparison_two_sided',
    required_terms_both_sides: { side_a: ['Xô viết', 'Nghệ Tĩnh'], side_b: ['Cần Vương'] },
    expected_citation_behavior: 'allow_multiple' },

  // ══════ DISAMBIGUATION (6) ══════
  { id: 'rb_disambig_001', category: 'disambiguation', difficulty: 'hard',
    query: 'Sự kiện nào nói về việc ký Hiệp định Paris, khác với việc Mỹ rút quân?',
    expected_behavior: 'Focus on signing event, not withdrawal',
    required_terms_any: ['ký', 'Paris', '1973'], expected_citation_behavior: 'at_least_one' },
  { id: 'rb_disambig_002', category: 'disambiguation', difficulty: 'hard',
    query: 'Hiệp định Genève 1954 khác với hội nghị Genève 1954 thế nào?',
    expected_behavior: 'Clarify or note they are related',
    required_terms_any: ['Genève', '1954'], expected_citation_behavior: 'at_least_one' },
  { id: 'rb_disambig_003', category: 'disambiguation', difficulty: 'medium',
    query: 'Bình thường hóa quan hệ Việt-Mỹ có phải là gia nhập ASEAN không?',
    expected_behavior: 'Clarify these are different events',
    required_terms_any: ['không', 'khác'], expected_citation_behavior: 'at_least_one' },
  { id: 'rb_disambig_004', category: 'disambiguation', difficulty: 'hard',
    query: 'Phân biệt chiến thắng Điện Biên Phủ 1954 với Điện Biên Phủ trên không 1972.',
    expected_behavior: 'Distinguish the two events',
    required_terms_any: ['1954', '1972'], expected_citation_behavior: 'allow_multiple' },
  { id: 'rb_disambig_005', category: 'disambiguation', difficulty: 'medium',
    query: 'Mặt trận Việt Minh và Mặt trận Liên Việt có phải cùng một tổ chức không?',
    expected_behavior: 'Clarify relationship between the two fronts',
    required_terms_any: ['Việt Minh', 'Liên Việt'], expected_citation_behavior: 'at_least_one' },
  { id: 'rb_disambig_006', category: 'disambiguation', difficulty: 'hard',
    query: 'Hiệp ước Nhâm Tuất 1862 có phải Hiệp ước Patenôtre không?',
    expected_behavior: 'Distinguish the two treaties',
    required_terms_any: ['không', 'khác', '1862', '1884'], expected_citation_behavior: 'at_least_one' },

  // ══════ MISCONCEPTION CHECK (5) ══════
  { id: 'rb_miscon_001', category: 'misconception_check', difficulty: 'hard',
    query: 'Hồ Chí Minh trực tiếp chỉ huy chiến dịch Điện Biên Phủ đúng không?',
    expected_behavior: 'Clarify: Võ Nguyên Giáp chỉ huy, not HCM directly',
    expected_answer_style: 'misconception_correction',
    required_terms_any: ['Giáp', 'Võ Nguyên Giáp', 'không trực tiếp'],
    expected_citation_behavior: 'at_least_one' },
  { id: 'rb_miscon_002', category: 'misconception_check', difficulty: 'medium',
    query: 'Cách mạng tháng Tám 1945 là do Liên Xô giúp đỡ đúng không?',
    expected_behavior: 'Clarify the role of domestic forces',
    expected_answer_style: 'misconception_correction',
    required_terms_any: ['Việt Minh', 'nhân dân', 'tự lực', 'tháng Tám'],
    expected_citation_behavior: 'at_least_one' },
  { id: 'rb_miscon_003', category: 'misconception_check', difficulty: 'hard',
    query: 'Có phải Hiệp định Paris 1973 đã chấm dứt hoàn toàn chiến tranh Việt Nam?',
    expected_behavior: 'Clarify: war continued until 1975',
    expected_answer_style: 'misconception_correction',
    required_terms_any: ['1975', 'chưa', 'tiếp tục', 'chưa chấm dứt'],
    expected_citation_behavior: 'at_least_one' },
  { id: 'rb_miscon_004', category: 'misconception_check', difficulty: 'medium',
    query: 'Đổi Mới bắt đầu từ năm 1975 phải không?',
    expected_behavior: 'Correct: Đổi Mới started 1986',
    expected_answer_style: 'misconception_correction',
    required_terms_any: ['1986', 'Đổi Mới'],
    expected_citation_behavior: 'at_least_one' },
  { id: 'rb_miscon_005', category: 'misconception_check', difficulty: 'hard',
    query: 'Việt Nam gia nhập ASEAN trước khi bình thường hóa quan hệ với Mỹ đúng không?',
    expected_behavior: 'Check chronological order of both events (both 1995)',
    expected_answer_style: 'misconception_correction',
    required_terms_any: ['1995', 'ASEAN'],
    expected_citation_behavior: 'at_least_one' },

  // ══════ OUT OF SCOPE (5) ══════
  { id: 'rb_oos_001', category: 'out_of_scope', difficulty: 'easy',
    query: 'Trận Bạch Đằng năm 938 diễn ra thế nào?',
    expected_behavior: 'Refuse — pre-1858',
    should_refuse: true, expected_citation_behavior: 'no_citation_for_oos',
    required_terms_any: ['ngoài phạm vi', '1858', 'không nằm trong'] },
  { id: 'rb_oos_002', category: 'out_of_scope', difficulty: 'easy',
    query: 'Việt Nam xử lý đại dịch COVID-19 như thế nào?',
    expected_behavior: 'Refuse — post-2000',
    should_refuse: true, expected_citation_behavior: 'no_citation_for_oos',
    required_terms_any: ['ngoài phạm vi', '2000', 'không nằm trong'] },
  { id: 'rb_oos_003', category: 'out_of_scope', difficulty: 'easy',
    query: 'Chiến tranh thế giới thứ nhất ảnh hưởng gì đến châu Âu?',
    expected_behavior: 'Refuse — outside VN history scope',
    should_refuse: true, expected_citation_behavior: 'no_citation_for_oos',
    required_terms_any: ['ngoài phạm vi', 'không nằm trong', 'không có trong'] },
  { id: 'rb_oos_004', category: 'out_of_scope', difficulty: 'medium',
    query: 'Vua Quang Trung đánh đuổi quân Thanh năm nào?',
    expected_behavior: 'Refuse — pre-1858',
    should_refuse: true, expected_citation_behavior: 'no_citation_for_oos',
    required_terms_any: ['ngoài phạm vi', '1858', 'không nằm trong'] },
  { id: 'rb_oos_005', category: 'out_of_scope', difficulty: 'medium',
    query: 'Việt Nam gia nhập Liên Hợp Quốc năm 2007 đúng không?',
    expected_behavior: 'May refuse (post-2000 context) or clarify VN joined UN in 1977',
    required_terms_any: ['1977', 'ngoài phạm vi', '2000', 'không'] },

  // ══════ ADVERSARIAL / VAGUE (4) ══════
  { id: 'rb_adv_001', category: 'adversarial', difficulty: 'hard',
    query: 'Nó diễn ra khi nào?',
    expected_behavior: 'Must ask for clarification — no specific referent',
    expected_answer_style: 'clarification_needed',
    clarification_expected: true,
    expected_citation_behavior: 'no_citation_for_ambiguous',
    notes: 'Vague query with no context — answering a random event is FAIL' },
  { id: 'rb_adv_002', category: 'adversarial', difficulty: 'hard',
    query: 'Sự kiện đó có ý nghĩa gì?',
    expected_behavior: 'Must ask for clarification — no specific referent',
    expected_answer_style: 'clarification_needed',
    clarification_expected: true,
    expected_citation_behavior: 'no_citation_for_ambiguous',
    notes: 'Vague query with no referent — answering a random event is FAIL' },
  { id: 'rb_adv_003', category: 'adversarial', difficulty: 'hard',
    query: 'Ai thắng?',
    expected_behavior: 'Must ask for clarification — no specific referent',
    expected_answer_style: 'clarification_needed',
    clarification_expected: true,
    expected_citation_behavior: 'no_citation_for_ambiguous',
    notes: 'Ambiguous query — answering a random event is FAIL' },
  { id: 'rb_adv_004', category: 'adversarial', difficulty: 'hard',
    query: 'Hiệp định đó có công bằng không?',
    expected_behavior: 'Must ask for clarification or note subjectivity — no specific referent',
    expected_answer_style: 'clarification_needed',
    clarification_expected: true,
    expected_citation_behavior: 'no_citation_for_ambiguous',
    notes: 'Subjective/opinion question — answering definitively about a random treaty is FAIL' },
];

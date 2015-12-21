switch (f->c) {
	case 1: { NATIVESORT(f, I); break; }
	case 2: { templatesort<chunk2,I>(f); break; }
	case 3: { templatesort<chunk3,I>(f); break; }
	case 4: { templatesort<chunk4,I>(f); break; }
	default:;
}
